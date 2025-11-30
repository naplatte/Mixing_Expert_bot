import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import update_binary_counts, compute_binary_f1

# 通用专家训练器 - 支持任意专家模型的训练/验证/测试
class ExpertTrainer:
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
        
        # 梯度裁剪
        self.max_grad_norm = expert_config.get('max_grad_norm', None)

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
            # extract_fn可能返回2个值(旧版)或3个值(新版，带has_data标记)
            result = self.extract_fn(batch, self.device)
            if len(result) == 3:
                inputs, labels, has_data = result
            else:
                inputs, labels = result
                has_data = None

            self.optimizer.zero_grad()
            
            # 前向传播
            _, bot_prob = self.model(*inputs)
            
            # 计算损失
            loss = self.criterion(bot_prob, labels)
            loss.backward()

            # 梯度裁剪（如果配置中提供）
            if hasattr(self, 'max_grad_norm') and self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

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
        valid_total = 0  # 有效样本总数
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"[{self.name.upper()}] Epoch {epoch} [Val]")
            for batch in pbar:
                # extract_fn可能返回2个值或3个值
                result = self.extract_fn(batch, self.device)
                if len(result) == 3:
                    inputs, labels, has_data = result
                else:
                    inputs, labels = result
                    has_data = None

                _, bot_prob = self.model(*inputs)
                loss = self.criterion(bot_prob, labels)
                
                total_loss += loss.item()
                predictions = (bot_prob > 0.5).float()

                # 如果有has_data标记，只在有效样本上计算指标
                if has_data is not None:
                    valid_mask = has_data.bool().view(-1)
                    valid_preds = predictions[valid_mask]
                    valid_labels = labels[valid_mask]

                    correct += (valid_preds == valid_labels).sum().item()
                    valid_total += valid_labels.size(0)
                    total += labels.size(0)

                    if valid_labels.size(0) > 0:
                        update_binary_counts(valid_preds, valid_labels, counts)
                else:
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    valid_total += labels.size(0)
                    update_binary_counts(predictions, labels, counts)

                _, _, f1_running = compute_binary_f1(counts)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}',
                    'f1': f'{f1_running:.4f}'
                })
        
        # 计算最终指标
        avg_loss = total_loss / len(self.val_loader)
        acc = correct / valid_total if valid_total > 0 else 0
        precision, recall, f1 = compute_binary_f1(counts)
        
        # 如果过滤了无效样本，显示统计信息
        if valid_total < total:
            print(f"  [验证集] 总样本: {total}, 有效样本: {valid_total} (过滤 {total - valid_total} 个无效样本)")

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
        valid_total = 0  # 有效样本总数
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"[{self.name.upper()}] Testing"):
                # extract_fn可能返回2个值或3个值
                result = self.extract_fn(batch, self.device)
                if len(result) == 3:
                    inputs, labels, has_data = result
                else:
                    inputs, labels = result
                    has_data = None

                _, bot_prob = self.model(*inputs)
                loss = self.criterion(bot_prob, labels)
                
                total_loss += loss.item()
                predictions = (bot_prob > 0.5).float()

                # 如果有has_data标记，只在有效样本上计算指标
                if has_data is not None:
                    valid_mask = has_data.bool().view(-1)
                    valid_preds = predictions[valid_mask]
                    valid_labels = labels[valid_mask]

                    correct += (valid_preds == valid_labels).sum().item()
                    valid_total += valid_labels.size(0)
                    total += labels.size(0)

                    if valid_labels.size(0) > 0:
                        update_binary_counts(valid_preds, valid_labels, counts)
                else:
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    valid_total += labels.size(0)
                    update_binary_counts(predictions, labels, counts)

        # 计算最终指标
        avg_loss = total_loss / len(self.test_loader)
        acc = correct / valid_total if valid_total > 0 else 0
        precision, recall, f1 = compute_binary_f1(counts)
        
        # 如果过滤了无效样本，显示统计信息
        if valid_total < total:
            print(f"  [测试集] 总样本: {total}, 有效样本: {valid_total} (过滤 {total - valid_total} 个无效样本)")

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

