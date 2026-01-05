"""
分层融合模块 - 实现 (Cat+Num) → (Des+Post) → 综合融合
完整实现包含训练、验证、测试功能，支持 ExpertTrainer 统一训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# 获取项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 添加到 Python 路径
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 分层融合数据集 - 支持缺失数据掩码
class HierarchicalFusionDataset(Dataset):
    def __init__(self, cat_embeddings, num_embeddings, des_embeddings, post_embeddings,
                 des_mask, post_mask, labels, mode='train'):
        """
        Args:
            cat_embeddings: 类别专家嵌入 [N, 64]
            num_embeddings: 数值专家嵌入 [N, 64]
            des_embeddings: 描述专家嵌入 [N, 64]
            post_embeddings: 推文专家嵌入 [N, 64]
            des_mask: 描述有效性掩码 [N] bool
            post_mask: 推文有效性掩码 [N] bool
            labels: 标签 [N]
            mode: 'train' | 'val' | 'test'
        """
        self.mode = mode
        self.cat_embeddings = cat_embeddings
        self.num_embeddings = num_embeddings
        self.des_embeddings = des_embeddings
        self.post_embeddings = post_embeddings
        self.des_mask = des_mask.bool() if isinstance(des_mask, torch.Tensor) else torch.tensor(des_mask, dtype=torch.bool)
        self.post_mask = post_mask.bool() if isinstance(post_mask, torch.Tensor) else torch.tensor(post_mask, dtype=torch.bool)
        self.labels = labels

        # 统计有效数据
        valid_des = self.des_mask.sum().item()
        valid_post = self.post_mask.sum().item()
        both_valid = (self.des_mask & self.post_mask).sum().item() # 同时有描述和推文的样本数量

        print(f"  [{mode}集] 样本数量: {len(self.labels)}")
        print(f"    有效描述: {valid_des}/{len(self.labels)} ({valid_des/len(self.labels)*100:.1f}%)")
        print(f"    有效推文: {valid_post}/{len(self.labels)} ({valid_post/len(self.labels)*100:.1f}%)")
        print(f"    两者都有: {both_valid}/{len(self.labels)} ({both_valid/len(self.labels)*100:.1f}%)")
        """
        [train集]
        样本数量: 8278
        有效描述: 7205 / 8278(87.0 %)
        有效推文: 8223 / 8278(99.3 %)
        两者都有: 7173 / 8278(86.7 %)
    
        [val集]
        样本数量: 2365
        有效描述: 2066 / 2365(87.4 %)
        有效推文: 2350 / 2365(99.4 %)
        两者都有: 2060 / 2365(87.1 %)

        [test集]
        样本数量: 1183
        有效描述: 1032 / 1183(87.2 %)
        有效推文: 1173 / 1183(99.2 %)
        两者都有: 1027 / 1183(86.8 %)
        """

    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
        return {
            'cat_repr': self.cat_embeddings[idx],
            'num_repr': self.num_embeddings[idx],
            'des_repr': self.des_embeddings[idx],
            'post_repr': self.post_embeddings[idx],
            'des_mask': self.des_mask[idx],
            'post_mask': self.post_mask[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class HierarchicalFusionExpert(nn.Module):
    """
    分层融合专家模型
    1. 第一层：属性融合 (Cat + Num) -> Property Representation (64d)
    2. 第二层：内容融合 (Des + Post) -> Content Representation (64d)
    3. 第三层：综合融合 (Property + Content) -> Final Representation (64d)
    4. 分类头：Final Representation -> Bot Probability

    特点：
    - 支持缺失数据掩码（Des/Post可能缺失）
    - 注意力机制自适应融合
    - 层归一化提升稳定性
    """

    def __init__(self, expert_dim=64, dropout=0.2, device='cuda'):
        super(HierarchicalFusionExpert, self).__init__()

        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.expert_dim = expert_dim

        # ========== 第一层：属性融合 (Cat + Num) ==========
        self.property_fusion = nn.Sequential(
            nn.Linear(expert_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        # ========== 第二层：内容融合 (Des + Post) ==========
        # 支持缺失数据的融合
        self.content_fusion = nn.Sequential(
            nn.Linear(expert_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        # 当Des或Post缺失时，单模态编码器
        self.des_encoder = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        self.post_encoder = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        # ========== 第三层：直接拼接融合（不使用注意力机制）==========
        # 直接拼接 property_repr (64d) + content_repr (64d) = 128d，然后MLP降维到64d
        self.final_fusion = nn.Sequential(
            nn.Linear(expert_dim * 2, 128),  # 128d -> 128d
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, expert_dim),      # 128d -> 64d
            nn.LayerNorm(expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        # ========== 分类头 ==========
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, cat_repr, num_repr, des_repr, post_repr, des_mask, post_mask):
        """
        Args:
            cat_repr: [B, 64] 类别专家表示
            num_repr: [B, 64] 数值专家表示
            des_repr: [B, 64] 描述专家表示
            post_repr: [B, 64] 推文专家表示
            des_mask: [B] bool, True表示有有效描述
            post_mask: [B] bool, True表示有有效推文

        Returns:
            final_repr: [B, 64] 最终融合表示
            bot_prob: [B, 1] bot概率
        """
        batch_size = cat_repr.shape[0]

        # ========== 第一层：类别数据 + 数值数据（拼接+降维）==========
        property_repr = self.property_fusion(
            torch.cat([cat_repr, num_repr], dim=1)
        )  # [B, 64]

        # ========== 第二层：简介文本 + 推文文本 ==========
        content_repr = torch.zeros(batch_size, self.expert_dim, device=self.device)

        # 情况1：Des和Post都有效（拼接 + MLP降维 128->64）
        both_valid = des_mask & post_mask
        if both_valid.any():
            content_repr[both_valid] = self.content_fusion(
                torch.cat([des_repr[both_valid], post_repr[both_valid]], dim=1)
            )

        # 情况2：只有Des有效
        only_des = des_mask & (~post_mask)
        if only_des.any():
            content_repr[only_des] = self.des_encoder(des_repr[only_des])

        # 情况3：只有Post有效
        only_post = (~des_mask) & post_mask
        if only_post.any():
            content_repr[only_post] = self.post_encoder(post_repr[only_post])

        # 情况4：都无效 - content_repr保持为零向量（已初始化）

        # ========== 第三层：直接拼接融合 ==========
        # 直接拼接 property_repr 和 content_repr，然后通过 MLP 降维
        concat = torch.cat([property_repr, content_repr], dim=1)  # [B, 128]
        final_repr = self.final_fusion(concat)  # [B, 64]

        # ========== 分类预测 ==========
        bot_prob = self.bot_classifier(final_repr)  # [B, 1]

        # ✅ 只返回2个值，与其他专家保持一致
        return final_repr, bot_prob

    def forward_with_attention(self, cat_repr, num_repr, des_repr, post_repr, des_mask, post_mask):
        """
        带注意力权重的前向传播（用于分析）
        注意：当前版本使用直接拼接而非注意力机制，返回的权重为均等权重以保持接口兼容

        Returns:
            final_repr: [B, 64] 最终融合表示
            bot_prob: [B, 1] bot概率
            attn_weights: [B, 2] 均等权重 (0.5, 0.5) - 保持接口兼容
        """
        batch_size = cat_repr.shape[0]

        # 第一层：属性融合
        property_repr = self.property_fusion(
            torch.cat([cat_repr, num_repr], dim=1)
        )

        # 第二层：内容融合
        content_repr = torch.zeros(batch_size, self.expert_dim, device=self.device)

        both_valid = des_mask & post_mask
        if both_valid.any():
            content_repr[both_valid] = self.content_fusion(
                torch.cat([des_repr[both_valid], post_repr[both_valid]], dim=1)
            )

        only_des = des_mask & (~post_mask)
        if only_des.any():
            content_repr[only_des] = self.des_encoder(des_repr[only_des])

        only_post = (~des_mask) & post_mask
        if only_post.any():
            content_repr[only_post] = self.post_encoder(post_repr[only_post])

        # 第三层：直接拼接融合
        concat = torch.cat([property_repr, content_repr], dim=1)
        final_repr = self.final_fusion(concat)

        bot_prob = self.bot_classifier(final_repr)

        # 返回均等权重以保持接口兼容
        attn_weights = torch.ones(batch_size, 2, device=self.device) * 0.5

        # 返回3个值（包含注意力权重）
        return final_repr, bot_prob, attn_weights

    def get_fused_repr(self, cat_repr, num_repr, des_repr, post_repr, des_mask, post_mask):
        """只获取融合后的表示"""
        final_repr, _ = self.forward(cat_repr, num_repr, des_repr, post_repr, des_mask, post_mask)
        return final_repr


def create_hierarchical_fusion_config(
        cat_embeddings_path='../../autodl-fs/labeled_embedding/cat_embeddings.pt',
        num_embeddings_path='../../autodl-fs/labeled_embedding/num_embeddings.pt',
        des_embeddings_path='../../autodl-fs/labeled_embedding/des_embeddings.pt',
        post_embeddings_path='../../autodl-fs/labeled_embedding/post_embeddings.pt',
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=0.01,
        device='cuda',
        checkpoint_dir='../../autodl-fs/model',
        dropout=0.2,
        early_stopping_patience=5,
        expert_dim=64,
        **kwargs
):
    """
    创建分层融合配置

    Args:
        cat_embeddings_path: 类别专家嵌入文件路径
        num_embeddings_path: 数值专家嵌入文件路径
        des_embeddings_path: 描述专家嵌入文件路径
        post_embeddings_path: 推文专家嵌入文件路径
        其他参数同前

    Returns:
        dict: 包含模型、数据加载器、优化器等的配置字典
    """
    print(f"\n{'='*60}")
    print(f"配置 分层融合模型 (Cat+Num → Des+Post → Final)")
    print(f"{'='*60}")

    # 检查嵌入文件是否存在
    required_files = {
        'cat': Path(cat_embeddings_path),
        'num': Path(num_embeddings_path),
        'des': Path(des_embeddings_path),
        'post': Path(post_embeddings_path)
    }

    missing_files = [name for name, path in required_files.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"\n缺少专家嵌入文件，请先训练以下专家：{', '.join(missing_files)}\n"
            f"训练命令: python configs/train_experts.py --expert {','.join(missing_files)} --num_epochs 10"
        )

    print(f"  类别专家嵌入: {cat_embeddings_path}")
    print(f"  数值专家嵌入: {num_embeddings_path}")
    print(f"  描述专家嵌入: {des_embeddings_path}")
    print(f"  推文专家嵌入: {post_embeddings_path}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  Dropout: {dropout}")

    # 加载所有专家嵌入
    print("\n加载专家嵌入...")
    cat_data = torch.load(cat_embeddings_path, map_location='cpu')
    num_data = torch.load(num_embeddings_path, map_location='cpu')
    des_data = torch.load(des_embeddings_path, map_location='cpu')
    post_data = torch.load(post_embeddings_path, map_location='cpu')

    # 准备训练、验证、测试数据
    def prepare_split_data(split_name):
        """准备指定split的数据"""
        cat_emb = cat_data[split_name]['embeddings']
        num_emb = num_data[split_name]['embeddings']
        des_emb = des_data[split_name]['embeddings']
        post_emb = post_data[split_name]['embeddings']

        # 获取mask（Des和Post可能有mask字段）
        des_mask = des_data[split_name].get('mask', torch.ones(len(des_emb), dtype=torch.bool))
        post_mask = post_data[split_name].get('mask', torch.ones(len(post_emb), dtype=torch.bool))

        # 标签（从任一数据源获取）
        labels = cat_data[split_name]['labels'].squeeze().numpy()

        return cat_emb, num_emb, des_emb, post_emb, des_mask, post_mask, labels

    # 训练集
    train_cat, train_num, train_des, train_post, train_des_mask, train_post_mask, train_labels = \
        prepare_split_data('train')

    # 验证集
    val_cat, val_num, val_des, val_post, val_des_mask, val_post_mask, val_labels = \
        prepare_split_data('val')

    # 测试集
    test_cat, test_num, test_des, test_post, test_des_mask, test_post_mask, test_labels = \
        prepare_split_data('test')

    # 验证数据一致性
    assert train_cat.shape[0] == train_num.shape[0] == train_des.shape[0] == train_post.shape[0], \
        "训练集样本数不一致"
    assert train_cat.shape[1] == expert_dim, f"嵌入维度错误: {train_cat.shape[1]} != {expert_dim}"

    print(f"  训练集: {len(train_labels)} 样本")
    print(f"  验证集: {len(val_labels)} 样本")
    print(f"  测试集: {len(test_labels)} 样本")
    print(f"  嵌入维度: {expert_dim}")

    # 创建数据集和数据加载器
    print("\n创建数据加载器...")
    train_dataset = HierarchicalFusionDataset(
        train_cat, train_num, train_des, train_post,
        train_des_mask, train_post_mask, train_labels, mode='train'
    )
    val_dataset = HierarchicalFusionDataset(
        val_cat, val_num, val_des, val_post,
        val_des_mask, val_post_mask, val_labels, mode='val'
    )
    test_dataset = HierarchicalFusionDataset(
        test_cat, test_num, test_des, test_post,
        test_des_mask, test_post_mask, test_labels, mode='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    print("\n初始化分层融合模型...")
    model = HierarchicalFusionExpert(
        expert_dim=expert_dim,
        dropout=dropout,
        device=device
    ).to(device)
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失函数
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    criterion = nn.BCELoss()

    # 数据提取函数
    def extract_fn(batch, device):
        cat_repr = batch['cat_repr'].to(device)
        num_repr = batch['num_repr'].to(device)
        des_repr = batch['des_repr'].to(device)
        post_repr = batch['post_repr'].to(device)
        des_mask = batch['des_mask'].to(device)
        post_mask = batch['post_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        return (cat_repr, num_repr, des_repr, post_repr, des_mask, post_mask), labels

    return {
        'name': 'hierarchical_fusion',
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'checkpoint_dir': checkpoint_dir,
        'extract_fn': extract_fn,
        'early_stopping_patience': early_stopping_patience,
        'max_grad_norm': 1.0,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'expert_dim': expert_dim
    }


def main():
    """
    主函数 - 使用 ExpertTrainer 统一训练
    """
    import argparse

    # 导入统一的训练器和metrics
    try:
        from expert_trainer import ExpertTrainer
        from src.metrics import update_binary_counts, compute_binary_f1
    except ImportError:
        print("错误: 无法导入 ExpertTrainer 或 metrics")
        print("请确保在项目根目录运行，或检查模块路径")
        return

    parser = argparse.ArgumentParser(description='训练分层融合模型')

    # 嵌入文件参数
    parser.add_argument('--cat_embeddings', type=str,
                        default='../../autodl-fs/labeled_embedding/cat_embeddings.pt',
                        help='类别专家嵌入文件路径')
    parser.add_argument('--num_embeddings', type=str,
                        default='../../autodl-fs/labeled_embedding/num_embeddings.pt',
                        help='数值专家嵌入文件路径')
    parser.add_argument('--des_embeddings', type=str,
                        default='../../autodl-fs/labeled_embedding/des_embeddings.pt',
                        help='描述专家嵌入文件路径')
    parser.add_argument('--post_embeddings', type=str,
                        default='../../autodl-fs/labeled_embedding/post_embeddings.pt',
                        help='推文专家嵌入文件路径')

    # 训练参数
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/model',
                        help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda 或 cpu)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout比率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--expert_dim', type=int, default=64,
                        help='专家嵌入维度')

    args = parser.parse_args()

    # 确定设备
    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 创建配置
    config = create_hierarchical_fusion_config(
        cat_embeddings_path=args.cat_embeddings,
        num_embeddings_path=args.num_embeddings,
        des_embeddings_path=args.des_embeddings,
        post_embeddings_path=args.post_embeddings,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        dropout=args.dropout,
        expert_dim=args.expert_dim
    )

    # 使用 ExpertTrainer 训练
    print("\n" + "=" * 60)
    print("使用 ExpertTrainer 训练分层融合模型")
    print("=" * 60)

    trainer = ExpertTrainer(config)
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_embeddings=True,
        embeddings_dir='../../autodl-fs/labeled_embedding/fusion'
    )

    print(f"\n{'=' * 60}")
    print(f"训练完成！")
    print(f"{'=' * 60}")
    print(f"最终测试结果:")
    for key, value in history['test'].items():
        print(f"  {key.capitalize()}: {value:.4f}")


if __name__ == '__main__':
    main()