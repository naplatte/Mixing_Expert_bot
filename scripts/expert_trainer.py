"""通用专家训练器"""
import torch
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import update_binary_counts, compute_binary_f1


class ExpertTrainer:
    """通用专家训练器"""

    def __init__(self, config):
        self.name = config['name']
        self.model = config['model']
        self.train_loader = config['train_loader']
        self.val_loader = config['val_loader']
        self.test_loader = config['test_loader']
        self.optimizer = config['optimizer']
        self.criterion = config['criterion']
        self.device = config['device']
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.extract_fn = config.get('extract_fn', self._default_extract)
        self.max_grad_norm = config.get('max_grad_norm')
        self.patience = config.get('early_stopping_patience')
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.history = {'train': [], 'val': [], 'test': {}}

    def _default_extract(self, batch, device):
        inputs = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        return inputs, labels

    def _run_epoch(self, loader, mode='train', epoch=0):
        """通用训练/验证/测试循环"""
        is_train = (mode == 'train')
        self.model.train() if is_train else self.model.eval()

        total_loss, correct, total = 0, 0, 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            desc = f"[{self.name.upper()}] Epoch {epoch} [{mode.capitalize()}]" if epoch else f"[{self.name.upper()}] {mode.capitalize()}"
            pbar = tqdm(loader, desc=desc)

            for batch in pbar:
                result = self.extract_fn(batch, self.device)
                inputs, labels = result[:2]

                if is_train:
                    self.optimizer.zero_grad()

                _, bot_prob = self.model(*inputs)
                loss = self.criterion(bot_prob, labels)

                if is_train:
                    loss.backward()
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_loss += loss.item()
                preds = (bot_prob > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                update_binary_counts(preds, labels, counts)

                _, _, f1_run = compute_binary_f1(counts)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}', 'f1': f'{f1_run:.4f}'})

        avg_loss = total_loss / len(loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        return {'loss': avg_loss, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def train_epoch(self, epoch):
        return self._run_epoch(self.train_loader, 'train', epoch)

    def validate(self, epoch):
        metrics = self._run_epoch(self.val_loader, 'val', epoch)

        # Graph专家不在训练过程中保存最优模型，只保存最终模型
        if self.name == 'graph':
            return metrics

        if metrics['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['loss']
            self._save_checkpoint('best', epoch, metrics)
            print(f"  ✓ 保存最佳模型 (Val F1: {metrics['f1']:.4f})")
            self.patience_counter = 0
        elif self.patience:
            self.patience_counter += 1
            print(f"  早停计数: {self.patience_counter}/{self.patience}")

        return metrics

    def test(self):
        print(f"\n[{self.name.upper()}] 测试...")
        # Graph专家不加载best模型，直接使用最后一轮的模型
        if self.name != 'graph':
            self._load_checkpoint('best')
        return self._run_epoch(self.test_loader, 'test')

    def train(self, num_epochs, save_embeddings=True, embeddings_dir='../../autodl-fs/labeled_embedding'):
        """完整训练流程"""
        print(f"\n{'='*60}\n开始训练 {self.name.upper()} Expert\n{'='*60}")

        for epoch in range(1, num_epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)

            self.history['train'].append(train_m)
            self.history['val'].append(val_m)

            print(f"Epoch {epoch}: Train F1={train_m['f1']:.4f}, Val F1={val_m['f1']:.4f}")

            if self.patience and self.patience_counter >= self.patience:
                print(f"早停触发！")
                break

        test_m = self.test()
        self.history['test'] = test_m

        print(f"\n{'='*60}\n{self.name.upper()} 测试结果:\n{'='*60}")
        print(f"  Acc: {test_m['acc']:.4f}, F1: {test_m['f1']:.4f}")
        print(f"  Precision: {test_m['precision']:.4f}, Recall: {test_m['recall']:.4f}")

        self._save_checkpoint('final', num_epochs, test_m)

        if save_embeddings:
            self.extract_and_save_embeddings(embeddings_dir)

        return self.history

    def extract_and_save_embeddings(self, save_dir='../../autodl-fs/labeled_embedding', force=False):
        """提取并保存embedding"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / f'{self.name}_embeddings.pt'

        if save_file.exists() and not force:
            print(f"[{self.name.upper()}] ✓ 嵌入文件已存在")
            return torch.load(save_file, map_location='cpu')

        print(f"\n[{self.name.upper()}] 提取embedding...")
        self._load_checkpoint('best')
        self.model.eval()

        all_embeddings = {}
        for split_name, loader in [('train', self.train_loader), ('val', self.val_loader), ('test', self.test_loader)]:
            emb_list, prob_list, label_list = [], [], []

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"[{self.name.upper()}] 提取 {split_name}"):
                    result = self.extract_fn(batch, self.device)
                    inputs, labels = result[:2]
                    expert_repr, bot_prob = self.model(*inputs)

                    emb_list.append(expert_repr.cpu())
                    prob_list.append(bot_prob.cpu())
                    label_list.append(labels.cpu())

            all_embeddings[split_name] = {
                'embeddings': torch.cat(emb_list, dim=0),
                'probs': torch.cat(prob_list, dim=0),
                'labels': torch.cat(label_list, dim=0)
            }
            print(f"  {split_name}: {all_embeddings[split_name]['embeddings'].shape}")

        torch.save(all_embeddings, save_file)
        print(f"  ✓ 保存到: {save_file}")
        return all_embeddings

    def _save_checkpoint(self, name, epoch, metrics):
        path = self.checkpoint_dir / f"{self.name}_expert_{name}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }, path)

    def _load_checkpoint(self, name):
        path = self.checkpoint_dir / f"{self.name}_expert_{name}.pt"
        if not path.exists():
            print(f"警告: 检查点 {path} 不存在")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
