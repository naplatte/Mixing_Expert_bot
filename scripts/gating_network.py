"""
门控网络 - 单文件实现
包含: 模型定义、数据加载、训练、验证、测试
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.dataset import Twibot20
from src.metrics import update_binary_counts, compute_binary_f1
from configs.expert_configs import get_expert_config


# ==================== 1. 模型定义 ====================
class GatingNetwork(nn.Module):
    """门控网络: 融合多个专家"""

    def __init__(self, num_experts=2, expert_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim

        # 注意力机制计算权重
        input_dim = num_experts * expert_dim
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(expert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, expert_embeddings, mask=None):
        """
        Args:
            expert_embeddings: List[Tensor(B,64)]
            mask: Tensor(B, num_experts) bool, True表示该专家可用
        Returns:
            final_prob: (B,1)
            weights: (B,num_experts)
        """
        batch_size = expert_embeddings[0].size(0)

        # 拼接所有专家嵌入
        concat = torch.cat(expert_embeddings, dim=1)  # (B, num_experts*64)

        # 计算注意力权重
        weights = self.attention(concat)  # (B, num_experts)

        # 如果提供了 mask，只激活有效专家
        if mask is not None:
            mask = mask.to(weights.device)
            # 对无效专家的权重设为 0
            weights = weights * mask.float()
            # 重新归一化（只对有效专家）
            weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights = weights / weights_sum

        # 加权融合
        stacked = torch.stack(expert_embeddings, dim=1)  # (B, num_experts, 64)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)  # (B, 64)

        # 最终分类
        final_prob = self.classifier(fused)  # (B, 1)

        return final_prob, weights


# ==================== 2. 数据集 ====================
class GatingDataset(Dataset):
    """门控网络数据集（支持动态专家激活）"""

    def __init__(self, expert_embeddings_dict, expert_masks_dict, labels):
        """
        Args:
            expert_embeddings_dict: {'des': Tensor(N,64), 'tweets': Tensor(N,64)}
            expert_masks_dict: {'des': Tensor(N,) bool, 'tweets': Tensor(N,) bool}
            labels: Tensor(N,)
        """
        self.expert_names = sorted(expert_embeddings_dict.keys())
        self.embeddings = expert_embeddings_dict
        self.masks = expert_masks_dict
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embeddings': [self.embeddings[name][idx] for name in self.expert_names],
            'mask': torch.tensor([self.masks[name][idx].item() for name in self.expert_names],
                                dtype=torch.bool),  # [num_experts,] 哪些专家可用
            'label': self.labels[idx].float()  # 转换为 float，BCELoss 需要
        }


def collate_fn(batch):
    """整理batch（包含 mask）"""
    num_experts = len(batch[0]['embeddings'])
    embeddings = [
        torch.stack([item['embeddings'][i] for item in batch])
        for i in range(num_experts)
    ]
    masks = torch.stack([item['mask'] for item in batch])  # [B, num_experts]
    labels = torch.stack([item['label'] for item in batch])
    return {'embeddings': embeddings, 'mask': masks, 'label': labels}


# ==================== 3. 提取专家特征（支持动态激活）====================
def extract_expert_features_with_mask(expert_name, split_indices, dataset_path, checkpoint_dir, device):
    """
    提取专家的64d嵌入，对于没有特征的样本用零向量填充，并返回可用性mask

    Args:
        expert_name: 'des' | 'tweets'
        split_indices: 样本索引列表
        dataset_path: 数据集路径
        checkpoint_dir: checkpoint目录
        device: 运行设备

    Returns:
        embeddings: Tensor(N, 64) - 所有样本的嵌入（缺失特征用零填充）
        mask: Tensor(N,) bool - True表示该样本有这个特征
    """
    print(f"  提取 {expert_name} expert 特征（支持动态激活）...")

    # 加载原始数据
    from src.dataset import Twibot20
    twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)

    if expert_name == 'des':
        raw_data = twibot_dataset.Des_preprocess()
    elif expert_name == 'tweets':
        raw_data = twibot_dataset.tweets_preprogress()
    else:
        raise ValueError(f"Unknown expert: {expert_name}")

    # 检查哪些样本有特征
    mask = torch.zeros(len(split_indices), dtype=torch.bool)
    valid_indices = []

    for i, idx in enumerate(split_indices):
        data = raw_data[idx]
        has_feature = False

        if expert_name == 'des':
            desc_str = str(data).strip()
            has_feature = (desc_str != '' and desc_str.lower() != 'none')
        elif expert_name == 'tweets':
            if isinstance(data, list):
                cleaned = [str(t).strip() for t in data if str(t).strip() not in ['', 'None']]
                has_feature = len(cleaned) > 0

        if has_feature:
            mask[i] = True
            valid_indices.append(i)

    print(f"    有效样本: {mask.sum().item()}/{len(split_indices)}")

    # 初始化全零嵌入
    embeddings = torch.zeros(len(split_indices), 64)

    # 如果有有效样本，提取特征
    if len(valid_indices) > 0:
        config = get_expert_config(
            expert_name,
            dataset_path=dataset_path,
            device=device,
            checkpoint_dir=checkpoint_dir
        )

        model = config['model']
        data_loader = config[f'train_loader']  # 仅用于获取 extract_fn
        extract_fn = config['extract_fn']

        # 加载最佳模型
        checkpoint_path = Path(checkpoint_dir) / f'{expert_name}_expert_best.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"    ✓ 已加载: {checkpoint_path}")

        model.eval()

        # 只对有效样本提取特征
        with torch.no_grad():
            # 构建临时数据集（只包含有效样本）
            valid_data = [raw_data[split_indices[i]] for i in valid_indices]
            labels_dummy = torch.zeros(len(valid_indices))

            # 根据专家类型创建数据集
            if expert_name == 'des':
                from configs.expert_configs import DescriptionDataset
                temp_dataset = DescriptionDataset(valid_data, labels_dummy.numpy(), mode='val')
            elif expert_name == 'tweets':
                from configs.expert_configs import TweetsDataset
                temp_dataset = TweetsDataset(valid_data, labels_dummy.numpy(), mode='val')

            # 为推文专家定义自定义 collate_fn（处理变长列表）
            def custom_collate(batch):
                if expert_name == 'tweets':
                    # tweets_text_list 是变长列表，不能直接 stack
                    return {
                        'tweets_text_list': [item['tweets_text'] for item in batch],
                        'label': torch.stack([item['label'] for item in batch])
                    }
                else:
                    # des 专家使用默认处理
                    return {
                        'description_text': [item['description_text'] for item in batch],
                        'label': torch.stack([item['label'] for item in batch])
                    }

            temp_loader = DataLoader(temp_dataset, batch_size=32, shuffle=False,
                                    collate_fn=custom_collate)

            embeddings_list = []
            for batch in tqdm(temp_loader, desc=f"    提取{expert_name}特征", leave=False):
                result = extract_fn(batch, device)
                if len(result) == 3:
                    inputs, _, _ = result
                else:
                    inputs, _ = result

                embedding, _ = model(*inputs)
                embeddings_list.append(embedding.cpu())

            valid_embeddings = torch.cat(embeddings_list, dim=0)

            # 将有效嵌入填充到对应位置
            for i, valid_emb in zip(valid_indices, valid_embeddings):
                embeddings[i] = valid_emb

    print(f"    特征形状: {embeddings.shape}, Mask: {mask.sum().item()} valid")

    return embeddings, mask


# ==================== 4. 训练器 ====================
class GatingTrainer:
    """门控网络训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, device, checkpoint_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            embeddings = [emb.to(self.device) for emb in batch['embeddings']]
            masks = batch['mask'].to(self.device)  # [B, num_experts]
            labels = batch['label'].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            final_prob, _ = self.model(embeddings, mask=masks)
            loss = self.criterion(final_prob, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions = (final_prob > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            update_binary_counts(predictions, labels, counts)
            _, _, f1 = compute_binary_f1(counts)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}',
                'f1': f'{f1:.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        return {'loss': avg_loss, 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                embeddings = [emb.to(self.device) for emb in batch['embeddings']]
                masks = batch['mask'].to(self.device)  # [B, num_experts]
                labels = batch['label'].to(self.device).unsqueeze(1)

                final_prob, _ = self.model(embeddings, mask=masks)
                loss = self.criterion(final_prob, labels)

                total_loss += loss.item()
                predictions = (final_prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                update_binary_counts(predictions, labels, counts)
                _, _, f1 = compute_binary_f1(counts)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}',
                    'f1': f'{f1:.4f}'
                })

        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': avg_loss,
                'val_acc': acc,
                'val_f1': f1
            }, self.checkpoint_dir / 'gating_best.pt')
            print(f"\n  ✓ 保存最佳模型 (Val Loss: {avg_loss:.4f}, F1: {f1:.4f})")

        return {'loss': avg_loss, 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    def test(self):
        """测试"""
        print("\n加载最佳模型进行测试...")
        checkpoint_path = self.checkpoint_dir / 'gating_best.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ 已加载: {checkpoint_path}")
        else:
            print(f"  ⚠ 未找到最佳模型，使用当前模型")

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                embeddings = [emb.to(self.device) for emb in batch['embeddings']]
                masks = batch['mask'].to(self.device)  # [B, num_experts]
                labels = batch['label'].to(self.device).unsqueeze(1)

                final_prob, _ = self.model(embeddings, mask=masks)
                loss = self.criterion(final_prob, labels)

                total_loss += loss.item()
                predictions = (final_prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                update_binary_counts(predictions, labels, counts)

        avg_loss = total_loss / len(self.test_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        return {'loss': avg_loss, 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    def train(self, num_epochs):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练门控网络")
        print("="*60 + "\n")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")

        # 测试
        print("\n" + "="*60)
        print("测试门控网络")
        print("="*60)
        test_metrics = self.test()

        print("\n" + "="*60)
        print("门控网络最终结果:")
        print("="*60)
        print(f"  Test Loss:      {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy:  {test_metrics['acc']:.4f}")
        print(f"  Test Precision: {test_metrics['precision']:.4f}")
        print(f"  Test Recall:    {test_metrics['recall']:.4f}")
        print(f"  Test F1 Score:  {test_metrics['f1']:.4f}")
        print("="*60 + "\n")



# ==================== 5. 主函数 ====================
def main():
    # ========== 配置 ==========
    expert_names = ['des', 'tweets']  # 使用的专家列表，在这里可以添加更多专家
    dataset_path = 'processed_data'
    checkpoint_dir = '../../autodl-tmp/checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-4

    print(f"\n{'='*60}")
    print(f"门控网络配置")
    print(f"{'='*60}")
    print(f"  使用专家: {expert_names}")
    print(f"  设备: {device}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率: {learning_rate}")

    # ========== 加载数据 ==========
    print("\n加载数据集...")
    twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    labels = twibot_dataset.load_labels().cpu()
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()

    print(f"  数据集划分:")
    print(f"    训练集索引数: {len(train_idx)}")
    print(f"    验证集索引数: {len(val_idx)}")
    print(f"    测试集索引数: {len(test_idx)}")

    # ========== 提取专家特征 ==========
    print("\n" + "="*60)
    print("提取专家特征（支持动态专家激活）")
    print("="*60)

    train_embeddings = {}
    train_masks = {}
    val_embeddings = {}
    val_masks = {}
    test_embeddings = {}
    test_masks = {}

    for expert_name in expert_names:
        print(f"\n处理 {expert_name} expert:")

        # 训练集
        train_emb, train_mask = extract_expert_features_with_mask(
            expert_name, train_idx, dataset_path, checkpoint_dir, device
        )
        train_embeddings[expert_name] = train_emb
        train_masks[expert_name] = train_mask

        # 验证集
        val_emb, val_mask = extract_expert_features_with_mask(
            expert_name, val_idx, dataset_path, checkpoint_dir, device
        )
        val_embeddings[expert_name] = val_emb
        val_masks[expert_name] = val_mask

        # 测试集
        test_emb, test_mask = extract_expert_features_with_mask(
            expert_name, test_idx, dataset_path, checkpoint_dir, device
        )
        test_embeddings[expert_name] = test_emb
        test_masks[expert_name] = test_mask

    # ========== 创建数据集 ==========
    print("\n" + "="*60)
    print("创建门控网络数据加载器")
    print("="*60)

    train_dataset = GatingDataset(train_embeddings, train_masks, labels[train_idx])
    val_dataset = GatingDataset(val_embeddings, val_masks, labels[val_idx])
    test_dataset = GatingDataset(test_embeddings, test_masks, labels[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")

    # ========== 初始化模型 ==========
    print("\n" + "="*60)
    print("初始化门控网络模型")
    print("="*60)

    model = GatingNetwork(
        num_experts=len(expert_names),
        expert_dim=64,
        hidden_dim=128,
        dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  门控网络参数量: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # ========== 训练 ==========
    trainer = GatingTrainer(
        model, train_loader, val_loader, test_loader,
        optimizer, criterion, device, checkpoint_dir
    )
    trainer.train(num_epochs)


if __name__ == '__main__':
    main()

