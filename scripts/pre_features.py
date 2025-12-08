"""
预计算专家特征 - 生成缓存文件
运行一次后，门控网络训练可直接使用缓存，大幅节省时间
"""
import torch
from pathlib import Path
import sys
import os
from tqdm import tqdm

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.dataset import Twibot20
from configs.expert_configs import get_expert_config


def extract_and_save_expert_features(expert_name, split_indices, split_name,
                                    dataset_path, checkpoint_dir, cache_dir, device,
                                    raw_data_dict, force=False):
    """提取并保存专家特征

    Args:
        expert_name: 专家名称 ('des' | 'tweets' | 'graph')
        split_indices: 样本索引列表
        split_name: 数据集划分名称 ('train' | 'val' | 'test')
        dataset_path: 数据集路径
        checkpoint_dir: checkpoint目录
        cache_dir: 缓存保存目录
        device: 运行设备
        raw_data_dict: 预加载的原始数据字典 {'des': [...], 'tweets': [...]}
        force: 是否强制重新提取（覆盖现有缓存）
    """
    # 检查缓存是否存在
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    save_file = cache_path / f'{expert_name}_{split_name}_features.pt'

    if save_file.exists() and not force:
        print(f"\n{'='*60}")
        print(f"跳过 {expert_name} expert - {split_name} 集（缓存已存在）")
        print(f"{'='*60}")
        print(f"  ✓ 缓存文件: {save_file}")

        # 加载并显示信息
        cached_data = torch.load(save_file, map_location='cpu')
        print(f"    特征形状: {cached_data['embeddings'].shape}, "
              f"有效样本: {cached_data['mask'].sum().item()}")
        return

    print(f"\n{'='*60}")
    print(f"提取 {expert_name} expert - {split_name} 集特征")
    print(f"{'='*60}")

    # 从预加载的数据字典中获取原始数据
    raw_data = raw_data_dict.get(expert_name)

    # 检查哪些样本有特征
    mask = torch.zeros(len(split_indices), dtype=torch.bool)
    valid_indices = []

    if expert_name == 'graph':
        # Graph Expert 对所有节点都可用
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

    print(f"  有效样本: {mask.sum().item()}/{len(split_indices)}")

    # 初始化全零嵌入
    embeddings = torch.zeros(len(split_indices), 64)

    # 如果有有效样本，提取特征
    if len(valid_indices) > 0:
        if expert_name == 'graph':
            # Graph Expert 特殊处理
            config = get_expert_config(
                expert_name,
                dataset_path=dataset_path,
                device=device,
                checkpoint_dir=checkpoint_dir,
                expert_names=['des', 'tweets']
            )

            model = config['model']
            edge_index = config['edge_index']
            edge_type = config['edge_type']

            # 加载最佳模型
            checkpoint_path = Path(checkpoint_dir) / f'{expert_name}_expert_best.pt'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✓ 已加载模型: {checkpoint_path}")
            else:
                print(f"  ⚠ 警告: 未找到 {checkpoint_path}，使用未训练的模型")

            model.eval()

            # 批量提取特征
            with torch.no_grad():
                node_indices = torch.tensor(split_indices, dtype=torch.long, device=device)
                batch_size = 256
                embeddings_list = []

                for i in tqdm(range(0, len(node_indices), batch_size),
                            desc=f"  提取特征", leave=False):
                    batch_indices = node_indices[i:i+batch_size]
                    batch_embeddings = model.get_expert_repr(batch_indices, edge_index, edge_type)
                    embeddings_list.append(batch_embeddings.cpu())

                embeddings = torch.cat(embeddings_list, dim=0)
        else:
            # Des 和 Tweets Expert 的处理
            config = get_expert_config(
                expert_name,
                dataset_path=dataset_path,
                device=device,
                checkpoint_dir=checkpoint_dir
            )

            model = config['model']
            extract_fn = config['extract_fn']

            # 加载最佳模型
            checkpoint_path = Path(checkpoint_dir) / f'{expert_name}_expert_best.pt'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✓ 已加载模型: {checkpoint_path}")
            else:
                print(f"  ⚠ 警告: 未找到 {checkpoint_path}，使用未训练的模型")

            model.eval()

            # 只对有效样本提取特征
            with torch.no_grad():
                valid_data = [raw_data[split_indices[i]] for i in valid_indices]
                labels_dummy = torch.zeros(len(valid_indices))

                # 创建临时数据集
                if expert_name == 'des':
                    from configs.expert_configs import DescriptionDataset
                    from torch.utils.data import DataLoader
                    temp_dataset = DescriptionDataset(valid_data, labels_dummy.numpy(), mode='val')

                    def collate_fn(batch):
                        return {
                            'description_text': [item['description_text'] for item in batch],
                            'label': torch.stack([item['label'] for item in batch])
                        }
                elif expert_name == 'tweets':
                    from configs.expert_configs import TweetsDataset
                    from torch.utils.data import DataLoader
                    temp_dataset = TweetsDataset(valid_data, labels_dummy.numpy(), mode='val')

                    def collate_fn(batch):
                        return {
                            'tweets_text_list': [item['tweets_text'] for item in batch],
                            'label': torch.stack([item['label'] for item in batch])
                        }

                temp_loader = DataLoader(temp_dataset, batch_size=32, shuffle=False,
                                       collate_fn=collate_fn)

                embeddings_list = []
                for batch in tqdm(temp_loader, desc=f"  提取特征", leave=False):
                    result = extract_fn(batch, device)
                    if len(result) == 3:
                        inputs, _, _ = result
                    else:
                        inputs, _ = result

                    embedding, _ = model(*inputs)
                    embeddings_list.append(embedding.cpu())

                valid_embeddings = torch.cat(embeddings_list, dim=0)

                # 填充到对应位置
                for i, valid_emb in zip(valid_indices, valid_embeddings):
                    embeddings[i] = valid_emb

    # 保存到缓存
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    save_file = cache_path / f'{expert_name}_{split_name}_features.pt'
    torch.save({
        'embeddings': embeddings,
        'mask': mask
    }, save_file)

    print(f"  ✓ 特征已保存: {save_file}")
    print(f"  特征形状: {embeddings.shape}, 有效样本: {mask.sum().item()}")


def main():
    """主函数：预计算所有专家的特征"""
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='预计算专家特征缓存')
    parser.add_argument('--force', action='store_true',
                       help='强制重新提取特征，覆盖现有缓存')
    parser.add_argument('--experts', type=str, default='des,tweets,graph',
                       help='要处理的专家列表（逗号分隔），默认: des,tweets,graph')
    parser.add_argument('--dataset_path', type=str, default='processed_data',
                       help='数据集路径')
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/checkpoints',
                       help='checkpoint目录')
    parser.add_argument('--cache_dir', type=str, default='../../autodl-fs/features',
                       help='特征缓存目录')
    args = parser.parse_args()

    # ========== 配置 ==========
    expert_names = [e.strip() for e in args.experts.split(',')]
    dataset_path = args.dataset_path
    checkpoint_dir = args.checkpoint_dir
    cache_dir = args.cache_dir
    force = args.force
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*60)
    print("预计算专家特征（一次性生成所有缓存）")
    print("="*60)
    print(f"  专家列表: {expert_names}")
    print(f"  设备: {device}")
    print(f"  缓存目录: {cache_dir}")
    print(f"  强制重新提取: {force}")

    # ========== 加载数据集（仅一次）==========
    print("\n加载数据集...")
    twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()

    print(f"  训练集: {len(train_idx)} 样本")
    print(f"  验证集: {len(val_idx)} 样本")
    print(f"  测试集: {len(test_idx)} 样本")

    # ========== 提取所有原始数据（仅一次）==========
    print("\n提取原始数据...")
    raw_data_dict = {}

    # 提取 description 数据
    print("  提取 description 数据...")
    des_data = []
    for i in range(len(twibot_dataset.df_data)):
        profile = twibot_dataset.df_data.iloc[i]['profile']
        if profile is None or profile['description'] is None:
            des_data.append('None')
        else:
            des_data.append(profile['description'])
    raw_data_dict['des'] = des_data

    # 提取 tweets 数据
    print("  提取 tweets 数据...")
    tweets_data = []
    for i in range(len(twibot_dataset.df_data)):
        tweet = twibot_dataset.df_data.iloc[i]['tweet']
        if tweet is None:
            tweets_data.append([''])
        else:
            tweets_data.append(tweet)
    raw_data_dict['tweets'] = tweets_data

    # Graph 数据不需要额外处理
    raw_data_dict['graph'] = None

    print(f"  ✓ 数据提取完成（总样本数: {len(des_data)}）")

    # ========== 预计算所有专家特征 ==========
    for expert_name in expert_names:
        print(f"\n{'#'*60}")
        print(f"处理专家: {expert_name}")
        print(f"{'#'*60}")

        # 训练集
        extract_and_save_expert_features(
            expert_name, train_idx, 'train',
            dataset_path, checkpoint_dir, cache_dir, device,
            raw_data_dict, force=force
        )

        # 验证集
        extract_and_save_expert_features(
            expert_name, val_idx, 'val',
            dataset_path, checkpoint_dir, cache_dir, device,
            raw_data_dict, force=force
        )

        # 测试集
        extract_and_save_expert_features(
            expert_name, test_idx, 'test',
            dataset_path, checkpoint_dir, cache_dir, device,
            raw_data_dict, force=force
        )

    print("\n" + "="*60)
    print("✓ 所有专家特征预计算完成！")
    print("="*60)
    print(f"缓存文件保存在: {cache_dir}")
    print("\n现在可以运行 gating_network.py，训练速度将大幅提升！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

