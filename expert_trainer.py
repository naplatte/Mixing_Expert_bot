"""
通用专家训练器
支持任意专家模型的训练/验证/测试，实现代码复用
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from pathlib import Path
from metrics import update_binary_counts, compute_binary_f1


class ExpertTrainer:
    """通用专家训练器 - 支持任意专家模型的训练/验证/测试"""
    
    def __init__(self, expert_config):
        """
        Args:
            expert_config: dict, 包含以下字段:
                - name: str, 专家名称 (如 'des', 'tweets', 'graph')
                - model: nn.Module, 专家模型
                - train_loader: DataLoader
                - val_loader: DataLoader
                - test_loader: DataLoader
                - optimizer: torch.optim
                - criterion: loss function
                - device: str
                - checkpoint_dir: str
                - extract_fn: function, 数据提取函数 (batch, device) -> (inputs, labels)
        """
        self.name = expert_config['name']
        self.model = expert_config['model']
        self.train_loader = expert_config['train_loader']
        self.val_loader = expert_config['val_loader']
        self.test_loader = expert_config['test_loader']
        self.optimizer = expert_config['optimizer']
        self.criterion = expert_config['criterion']
        self.device = expert_config['device']
        self.checkpoint_dir = Path(expert_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据提取函数(不同专家数据格式不同)
        self.extract_fn = expert_config.get('extract_fn', self._default_extract)
        
        self.best_val_loss = float('inf')
        self.history = {'train': [], 'val': [], 'test': {}}
    
    def _default_extract(self, batch, device):
        """默认数据提取函数"""
        inputs = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        return inputs, labels
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        pbar = tqdm(self.train_loader, desc=f"[{self.name.upper()}] Epoch {epoch} [Train]")
        for batch in pbar:
            # 提取数据(适配不同专家)
            inputs, labels = self.extract_fn(batch, self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            _, bot_prob = self.model(*inputs)
            
            # 计算损失
            loss = self.criterion(bot_prob, labels)
            loss.backward()
            self.optimizer.step()
            
            # 统计指标
            total_loss += loss.item()
            predictions = (bot_prob > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # 更新F1计数
            update_binary_counts(predictions, labels, counts)
            _, _, f1_running = compute_binary_f1(counts)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}',
                'f1': f'{f1_running:.4f}'
            })
        
        # 计算最终指标
        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)
        
        return {
            'loss': avg_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"[{self.name.upper()}] Epoch {epoch} [Val]")
            for batch in pbar:
                inputs, labels = self.extract_fn(batch, self.device)
                
                _, bot_prob = self.model(*inputs)
                loss = self.criterion(bot_prob, labels)
                
                total_loss += loss.item()
                predictions = (bot_prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                update_binary_counts(predictions, labels, counts)
                _, _, f1_running = compute_binary_f1(counts)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}',
                    'f1': f'{f1_running:.4f}'
                })
        
        # 计算最终指标
        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)
        
        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best', epoch, {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1})
            print(f"  ✓ 保存最佳模型 (Val Loss: {avg_loss:.4f}, Val F1: {f1:.4f})")
        
        return {
            'loss': avg_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def test(self):
        """测试"""
        print(f"\n[{self.name.upper()}] 开始测试...")
        
        # 加载最佳模型
        self.load_checkpoint('best')
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"[{self.name.upper()}] Testing"):
                inputs, labels = self.extract_fn(batch, self.device)
                
                _, bot_prob = self.model(*inputs)
                loss = self.criterion(bot_prob, labels)
                
                total_loss += loss.item()
                predictions = (bot_prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                update_binary_counts(predictions, labels, counts)
        
        # 计算最终指标
        avg_loss = total_loss / len(self.test_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)
        
        return {
            'loss': avg_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"开始训练 {self.name.upper()} Expert")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)
            
            # 打印epoch结果
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 测试
        test_metrics = self.test()
        self.history['test'] = test_metrics
        
        print(f"\n{'='*60}")
        print(f"{self.name.upper()} Expert 测试结果:")
        print(f"{'='*60}")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['acc']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"{'='*60}\n")
        
        # 保存最终模型和历史记录
        self.save_checkpoint('final', num_epochs, test_metrics)
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, name, epoch=None, extra_info=None):
        """保存检查点"""
        path = self.checkpoint_dir / f"{self.name}_expert_{name}.pt"
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, name):
        """加载检查点"""
        path = self.checkpoint_dir / f"{self.name}_expert_{name}.pt"
        if not path.exists():
            print(f"警告: 检查点 {path} 不存在，使用当前模型状态")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_history(self):
        """保存训练历史"""
        path = self.checkpoint_dir / f"{self.name}_expert_history.json"
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ 训练历史已保存到: {path}")

