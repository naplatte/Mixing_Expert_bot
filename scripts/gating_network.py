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
    """门控网络: 融合多个专家（先筛选有效专家，再计算权重）
    支持的专家: des, tweets, graph
    """

    def __init__(self, num_experts=3, expert_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim

        # 对每个专家打分（分数归一化后就是权重）
        self.expert_scorer = nn.Sequential(
            nn.Linear(expert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 输出单个 logit
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
            expert_embeddings: List[Tensor(B,64)] B表示batch size大小
            mask: Tensor(B, num_experts) bool, True表示该专家可用
        Returns:
            final_prob: (B,1) batch size中每个样本的最终预测概率
            weights: (B,num_experts) 每个专家的权重
        """
        batch_size = expert_embeddings[0].size(0)
        device = expert_embeddings[0].device

        # 堆叠所有专家嵌入（样本某数据为空则对应专家表示用零向量填充）
        stacked = torch.stack(expert_embeddings, dim=1)  # (B, num_experts, 64)

        if mask is None:
            # 如果没有 mask，所有专家都可用
            mask = torch.ones(batch_size, self.num_experts, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)

        # 初始化输出
        fused_list = []
        weights_list = []

        # 对每个样本单独处理（只使用其有效专家）
        for i in range(batch_size):
            sample_mask = mask[i]  # (num_experts,)
            valid_indices = torch.where(sample_mask)[0]  # 有效专家的索引

            if len(valid_indices) == 0:
                # 如果没有有效专家，使用零向量和均匀权重
                fused_list.append(torch.zeros(self.expert_dim, device=device))
                weights_list.append(torch.zeros(self.num_experts, device=device))
            else:
                # 只提取有效专家的嵌入
                valid_embeddings = stacked[i, valid_indices]  # (num_valid, 64)

                # 对有效专家打分
                logits = self.expert_scorer(valid_embeddings).squeeze(-1)  # (num_valid,)

                # Softmax 归一化（只在有效专家间）
                valid_weights = torch.softmax(logits, dim=0)  # (num_valid,)

                # 加权融合有效专家
                fused = torch.sum(valid_embeddings * valid_weights.unsqueeze(-1), dim=0)  # (64,)
                fused_list.append(fused)

                # 构建完整的权重向量（无效专家权重为0）
                full_weights = torch.zeros(self.num_experts, device=device)
                full_weights[valid_indices] = valid_weights
                weights_list.append(full_weights)

        # 合并batch
        fused = torch.stack(fused_list, dim=0)  # (B, 64)
        weights = torch.stack(weights_list, dim=0)  # (B, num_experts)

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


# ==================== 3. 提取专家特征（支持动态激活和缓存）====================
def save_expert_features(expert_name, embeddings, mask, split_name, cache_dir):
    """保存专家特征到磁盘

    Args:
        expert_name: 专家名称
        embeddings: Tensor(N, 64) - 特征嵌入
        mask: Tensor(N,) bool - 可用性mask
        split_name: 'train' | 'val' | 'test'
        cache_dir: 缓存目录路径
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    save_file = cache_path / f'{expert_name}_{split_name}_features.pt'
    torch.save({
        'embeddings': embeddings,
        'mask': mask
    }, save_file)
    print(f"    ✓ 已保存特征缓存: {save_file}")


def load_expert_features(expert_name, split_name, cache_dir):
    """从磁盘加载专家特征

    Args:
        expert_name: 专家名称
        split_name: 'train' | 'val' | 'test'
        cache_dir: 缓存目录路径

    Returns:
        embeddings: Tensor(N, 64) 或 None（如果缓存不存在）
        mask: Tensor(N,) bool 或 None
    """
    cache_file = Path(cache_dir) / f'{expert_name}_{split_name}_features.pt'
    if cache_file.exists():
        data = torch.load(cache_file)
        print(f"    ✓ 已加载特征缓存: {cache_file}")
        return data['embeddings'], data['mask']
    return None, None


def extract_expert_features_with_mask(expert_name, split_indices, dataset_path, checkpoint_dir, device, twibot_dataset=None):
    """
    提取专家的64d嵌入，对于没有特征的样本用零向量填充，并返回可用性mask

    Args:
        expert_name: 'des' | 'tweets' | 'graph'
        split_indices: 样本索引列表
        dataset_path: 数据集路径
        checkpoint_dir: checkpoint目录
        device: 运行设备
        twibot_dataset: 预加载的 Twibot20 数据集对象（可选，避免重复加载）

    Returns:
        embeddings: Tensor(N, 64) - 所有样本的嵌入（缺失特征用零填充）
        mask: Tensor(N,) bool - True表示该样本有这个特征
    """
    print(f"  提取 {expert_name} expert 特征（支持动态激活）...")

    # 如果没有提供预加载的数据集，则加载
    if twibot_dataset is None:
        from src.dataset import Twibot20
        twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)

    # 获取原始数据 - 注意：需要使用包含支持集的完整数据
    if expert_name == 'des':
        # Des_preprocess 默认只返回有标签数据，需要处理完整数据
        # 直接从 df_data 中提取描述信息（包含支持集）
        raw_data = []
        for i in range(len(twibot_dataset.df_data)):
            profile = twibot_dataset.df_data.iloc[i]['profile']
            if profile is None or profile['description'] is None:
                raw_data.append('None')
            else:
                raw_data.append(profile['description'])
    elif expert_name == 'tweets':
        # tweets_preprogress 同理，需要处理完整数据
        raw_data = []
        for i in range(len(twibot_dataset.df_data)):
            tweet = twibot_dataset.df_data.iloc[i]['tweet']
            if tweet is None:
                raw_data.append([''])
            else:
                raw_data.append(tweet)
    elif expert_name == 'graph':
        # Graph Expert 不需要 raw_data，因为图结构对所有节点都可用
        raw_data = None
    else:
        raise ValueError(f"Unknown expert: {expert_name}")

    # 检查哪些样本有特征
    mask = torch.zeros(len(split_indices), dtype=torch.bool)
    valid_indices = []

    if expert_name == 'graph':
        # Graph Expert 对所有节点都可用（因为图结构是全局的）
        mask = torch.ones(len(split_indices), dtype=torch.bool)
        valid_indices = list(range(len(split_indices)))
    else:
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
        if expert_name == 'graph':
            # Graph Expert 特殊处理：需要加载图结构
            config = get_expert_config(
                expert_name,
                dataset_path=dataset_path,
                device=device,
                checkpoint_dir=checkpoint_dir,
                expert_names=['des', 'tweets']  # Graph Expert 依赖的专家
            )

            model = config['model']
            edge_index = config['edge_index']
            edge_type = config['edge_type']

            # 加载最佳模型
            checkpoint_path = Path(checkpoint_dir) / f'{expert_name}_expert_best.pt'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"    ✓ 已加载: {checkpoint_path}")
            else:
                print(f"    ⚠ 警告: 未找到 {checkpoint_path}，使用未训练的模型")

            model.eval()

            # 批量提取特征（Graph Expert 可以一次处理所有节点）
            with torch.no_grad():
                # 将 split_indices 转换为 tensor
                node_indices = torch.tensor(split_indices, dtype=torch.long, device=device)

                # 分批处理以避免内存问题
                batch_size = 256
                embeddings_list = []

                for i in tqdm(range(0, len(node_indices), batch_size),
                             desc=f"    提取{expert_name}特征", leave=False):
                    batch_indices = node_indices[i:i+batch_size]
                    batch_embeddings = model.get_expert_repr(batch_indices, edge_index, edge_type)
                    embeddings_list.append(batch_embeddings.cpu())

                embeddings = torch.cat(embeddings_list, dim=0)
        else:
            # Des 和 Tweets Expert 的处理逻辑
            # 注意：这里只需要加载模型，不需要加载完整的config（避免重复加载数据集）
            from src.model import DesExpert, TweetsExpert

            if expert_name == 'des':
                model = DesExpert(model_name='microsoft/deberta-v3-base', device=device, dropout=0.2).to(device)
            elif expert_name == 'tweets':
                model = TweetsExpert(roberta_model_name='distilroberta-base', device=device).to(device)
            else:
                # 其他专家需要完整config
                config = get_expert_config(
                    expert_name,
                    dataset_path=dataset_path,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                    twibot_dataset=twibot_dataset
                )
                model = config['model']
                extract_fn = config['extract_fn']

            # 加载最佳模型
            checkpoint_path = Path(checkpoint_dir) / f'{expert_name}_expert_best.pt'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"    ✓ 已加载: {checkpoint_path}")

            model.eval()

            # 只对有效样本提取特征（直接调用模型的forward，不需要dataset/dataloader）
            with torch.no_grad():
                valid_data = [raw_data[split_indices[i]] for i in valid_indices]

                # 批量处理以提高效率
                batch_size = 32
                embeddings_list = []

                for i in tqdm(range(0, len(valid_data), batch_size),
                             desc=f"    提取{expert_name}特征", leave=False):
                    batch_data = valid_data[i:i+batch_size]

                    # 直接调用模型forward
                    if expert_name == 'des':
                        # DesExpert.forward 接受 List[str]
                        embedding, _ = model(batch_data)
                    elif expert_name == 'tweets':
                        # TweetsExpert.forward 接受 List[List[str]]
                        embedding, _ = model(batch_data)

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
    expert_names = ['des', 'tweets', 'graph']  # 使用的专家列表（现已包含图专家）
    dataset_path = 'processed_data'
    checkpoint_dir = '../../autodl-fs/checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    num_epochs = 20
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

    # ========== 提取专家特征（带缓存优化）==========
    cache_dir = '../../autodl-fs/features'
    use_cache = True  # 设置为 False 可强制重新提取特征

    print("\n" + "="*60)
    print("提取专家特征（支持缓存加速）")
    print("="*60)
    print(f"  缓存目录: {cache_dir}")
    print(f"  使用缓存: {use_cache}")

    train_embeddings = {}
    train_masks = {}
    val_embeddings = {}
    val_masks = {}
    test_embeddings = {}
    test_masks = {}

    for expert_name in expert_names:
        print(f"\n处理 {expert_name} expert:")

        # 训练集 - 尝试从缓存加载
        if use_cache:
            train_emb, train_mask = load_expert_features(expert_name, 'train', cache_dir)
        else:
            train_emb, train_mask = None, None

        if train_emb is None:
            print("  [训练集] 缓存未找到，重新提取特征...")
            train_emb, train_mask = extract_expert_features_with_mask(
                expert_name, train_idx, dataset_path, checkpoint_dir, device
            )
            save_expert_features(expert_name, train_emb, train_mask, 'train', cache_dir)

        train_embeddings[expert_name] = train_emb
        train_masks[expert_name] = train_mask

        # 验证集 - 尝试从缓存加载
        if use_cache:
            val_emb, val_mask = load_expert_features(expert_name, 'val', cache_dir)
        else:
            val_emb, val_mask = None, None

        if val_emb is None:
            print("  [验证集] 缓存未找到，重新提取特征...")
            val_emb, val_mask = extract_expert_features_with_mask(
                expert_name, val_idx, dataset_path, checkpoint_dir, device
            )
            save_expert_features(expert_name, val_emb, val_mask, 'val', cache_dir)

        val_embeddings[expert_name] = val_emb
        val_masks[expert_name] = val_mask

        # 测试集 - 尝试从缓存加载
        if use_cache:
            test_emb, test_mask = load_expert_features(expert_name, 'test', cache_dir)
        else:
            test_emb, test_mask = None, None

        if test_emb is None:
            print("  [测试集] 缓存未找到，重新提取特征...")
            test_emb, test_mask = extract_expert_features_with_mask(
                expert_name, test_idx, dataset_path, checkpoint_dir, device
            )
            save_expert_features(expert_name, test_emb, test_mask, 'test', cache_dir)

        test_embeddings[expert_name] = test_emb
        test_masks[expert_name] = test_mask

    print(f"\n✓ 所有专家特征准备完毕（{'使用缓存' if use_cache else '重新提取'}）")

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

