"""
统一专家训练入口
使用通用训练器训练所有专家模型
"""
import torch
import argparse
import sys
from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from expert_trainer import ExpertTrainer
from configs.expert_configs import get_expert_config

# 训练单个专家
def train_single_expert(expert_name, config_params, num_epochs=10, save_embeddings=True, embeddings_dir='../../autodl-fs/labeled_embedding'):
    # 根据专家类型调整 MoE 参数
    adjusted_params = config_params.copy()

    if expert_name == 'cat':
        # Cat Expert 使用专用的 MoE 参数（3选1）
        if 'cat_num_experts' in adjusted_params:
            adjusted_params['num_experts'] = adjusted_params['cat_num_experts']
        if 'cat_top_k' in adjusted_params:
            adjusted_params['top_k'] = adjusted_params['cat_top_k']

    elif expert_name == 'num':
        # Num Expert 使用专用的 MoE 参数（3选1）
        if 'num_num_experts' in adjusted_params:
            adjusted_params['num_experts'] = adjusted_params['num_num_experts']
        if 'num_top_k' in adjusted_params:
            adjusted_params['top_k'] = adjusted_params['num_top_k']

    elif expert_name == 'graph':
        # Graph Expert 使用专用的 MoE 参数（2选1）
        if 'graph_num_experts' in adjusted_params:
            adjusted_params['num_experts'] = adjusted_params['graph_num_experts']
        if 'graph_top_k' in adjusted_params:
            adjusted_params['top_k'] = adjusted_params['graph_top_k']
        if 'graph_dropout' in adjusted_params:
            adjusted_params['dropout'] = adjusted_params['graph_dropout']

    # 获取专家配置
    config = get_expert_config(expert_name, **adjusted_params)

    # 创建训练器
    trainer = ExpertTrainer(config)

    # 对于 des 和 post 专家，使用带 mask 的 embedding 提取方法
    if expert_name in ['des', 'post'] and save_embeddings:
        # 先训练（不保存 embedding）
        history = trainer.train(num_epochs, save_embeddings=False, embeddings_dir=embeddings_dir)
        
        # 然后使用带 mask 的方法提取 embedding
        # 需要加载原始数据和 split 索引
        from src.dataset import Twibot20
        import numpy as np
        
        dataset_path = adjusted_params.get('dataset_path', str(project_root / 'processed_data'))
        device = adjusted_params.get('device', 'cuda')
        
        twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
        
        # 获取原始数据
        if expert_name == 'des':
            raw_data = twibot_dataset.Des_preprocess()
        else:  # post
            raw_data = twibot_dataset.tweets_preprogress()
        
        if isinstance(raw_data, np.ndarray):
            raw_data = raw_data.tolist()
        
        # 获取 split 索引
        train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
        split_indices = {
            'train': list(train_idx),
            'val': list(val_idx),
            'test': list(test_idx)
        }
        
        # 提取带 mask 的 embedding
        trainer.extract_and_save_embeddings_with_mask(
            save_dir=embeddings_dir,
            raw_data_list=raw_data,
            split_indices=split_indices,
            force=True
        )
    else:
        # 其他专家使用原有方法
        history = trainer.train(num_epochs, save_embeddings=save_embeddings, embeddings_dir=embeddings_dir)

    return history


def check_expert_dependencies(expert_name, checkpoint_dir):
    """
    检查专家的依赖是否满足

    Args:
        expert_name: 专家名称
        checkpoint_dir: 检查点目录

    Returns:
        bool: 依赖是否满足
    """
    # 定义专家依赖关系
    dependencies = {
        'graph': ['des', 'post'],  # 图专家依赖 des 和 post
    }

    if expert_name not in dependencies:
        return True  # 没有依赖，直接返回True

    required_experts = dependencies[expert_name]
    checkpoint_dir = Path(checkpoint_dir)

    print(f"  检查 {expert_name} 专家的依赖...")
    for dep_expert in required_experts:
        checkpoint_path = checkpoint_dir / f'{dep_expert}_expert_best.pt'
        if not checkpoint_path.exists():
            print(f"    ✗ 缺少依赖: {dep_expert} 专家未训练")
            print(f"    请先训练 {dep_expert} 专家")
            return False
        else:
            print(f"    ✓ {dep_expert} 专家已训练")

    return True


def train_all_experts(config_params, num_epochs=10, experts=None, save_embeddings=True, embeddings_dir='../../autodl-fs/labeled_embedding'):
    """
    训练所有专家或指定的专家列表

    Args:
        config_params: 配置参数字典
        num_epochs: 训练轮数
        experts: 要训练的专家列表，None表示训练所有可用专家
        save_embeddings: 是否保存特征嵌入
        embeddings_dir: 特征嵌入保存目录

    Returns:
        dict: 所有专家的训练结果
    """
    # 默认训练的专家列表（按依赖顺序）
    if experts is None:
        experts = ['des', 'post', 'cat', 'num', 'graph']  # graph 依赖 des 和 post，所以放最后

    results = {}

    print("\n" + "="*60)
    print("开始训练所有专家模型")
    print("="*60)
    print(f"专家列表: {experts}")
    print(f"训练轮数: {num_epochs}")
    print(f"设备: {config_params['device']}")
    print(f"保存特征嵌入: {save_embeddings}")
    if save_embeddings:
        print(f"特征嵌入保存路径: {embeddings_dir}")
    print("="*60 + "\n")

    # 逐个训练专家
    for expert_name in experts:
        print(f"\n{'#'*60}")
        print(f"# 训练专家: {expert_name.upper()}")
        print(f"{'#'*60}\n")

        # 检查依赖
        if not check_expert_dependencies(expert_name, config_params['checkpoint_dir']):
            print(f"\n⚠ 跳过 {expert_name} 专家（依赖不满足）\n")
            continue

        try:
            history = train_single_expert(expert_name, config_params, num_epochs,
                                         save_embeddings=save_embeddings,
                                         embeddings_dir=embeddings_dir)
            results[expert_name] = history
        except Exception as e:
            print(f"\n错误: 训练 {expert_name} 专家时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 打印所有专家的对比结果
    print("\n" + "="*60)
    print("所有专家测试结果对比")
    print("="*60)
    print(f"{'专家':<15} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*60)

    for name, history in results.items():
        test = history.get('test', {})
        print(f"{name.upper():<15} "
              f"{test.get('acc', 0):<12.4f} "
              f"{test.get('f1', 0):<12.4f} "
              f"{test.get('precision', 0):<12.4f} "
              f"{test.get('recall', 0):<12.4f}")

    print("="*60 + "\n")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练社交机器人检测专家模型')
    parser.add_argument('--expert', type=str, default='all',
                        help='要训练的专家 (des, post, tweets, cat, num, graph, all)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='数据集路径')
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/model',
                        help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率 (None表示使用默认值)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda 或 cpu)')

    # Des Expert 和 Post Expert MoE 参数
    parser.add_argument('--num_experts', type=int, default=4,
                        help='Des/Post Expert MoE 中的专家数量 (默认4)')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Des/Post Expert MoE Top-K 选择数量 (默认2)')

    # Cat Expert MoE 参数
    parser.add_argument('--cat_num_experts', type=int, default=3,
                        help='Cat Expert MoE 中的专家数量 (默认3)')
    parser.add_argument('--cat_top_k', type=int, default=1,
                        help='Cat Expert MoE Top-K 选择数量 (默认1)')

    # Num Expert MoE 参数
    parser.add_argument('--num_num_experts', type=int, default=3,
                        help='Num Expert MoE 中的专家数量 (默认3)')
    parser.add_argument('--num_top_k', type=int, default=1,
                        help='Num Expert MoE Top-K 选择数量 (默认1)')

    # 旧版 Tweets Expert 参数 (保留兼容)
    parser.add_argument('--roberta_model', type=str, default='distilroberta-base',
                        help='RoBERTa模型名称 (用于旧版 Tweets Expert)')

    # Graph Expert 参数 (MoE 版本) - 2选1
    parser.add_argument('--graph_embedding_dim', type=int, default=128,
                        help='图专家RGCN隐藏层维度')
    parser.add_argument('--graph_num_experts', type=int, default=2,
                        help='图专家MoE中的专家数量 (2选1)')
    parser.add_argument('--graph_top_k', type=int, default=1,
                        help='图专家MoE Top-K选择数量')
    parser.add_argument('--graph_dropout', type=float, default=0.3,
                        help='图专家Dropout比率')

    # 特征嵌入保存参数
    parser.add_argument('--save_embeddings', action='store_true', default=True,
                        help='是否保存特征嵌入')
    parser.add_argument('--embeddings_dir', type=str, default='../../autodl-fs/labeled_embedding',
                        help='特征嵌入保存目录')

    args = parser.parse_args()

    # 确定设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # 解析数据集根目录，默认使用项目根目录下的 processed_data
    dataset_path = args.dataset_path or str(project_root / 'processed_data')

    print(f"\n使用设备: {device}")

    # 基础配置参数
    base_config = {
        'dataset_path': dataset_path,
        'batch_size': args.batch_size,
        'device': device,
        'checkpoint_dir': args.checkpoint_dir,
        # Des/Post Expert MoE 参数
        'num_experts': args.num_experts,
        'top_k': args.top_k,
        # Cat Expert MoE 参数 (3选1)
        'cat_num_experts': args.cat_num_experts,
        'cat_top_k': args.cat_top_k,
        # Num Expert MoE 参数 (3选1)
        'num_num_experts': args.num_num_experts,
        'num_top_k': args.num_top_k,
        # 旧版 Tweets Expert 参数 (兼容)
        'roberta_model_name': args.roberta_model,
        # Graph Expert 参数 (MoE 版本)
        'embedding_dim': args.graph_embedding_dim,
        'graph_num_experts': args.graph_num_experts,
        'graph_top_k': args.graph_top_k,
        'graph_dropout': args.graph_dropout,
    }

    # 添加学习率 (如果指定)
    if args.learning_rate is not None:
        base_config['learning_rate'] = args.learning_rate

    # 训练专家
    if args.expert == 'all':
        # 训练所有专家
        results = train_all_experts(base_config, args.num_epochs,
                                   save_embeddings=args.save_embeddings,
                                   embeddings_dir=args.embeddings_dir)
    else:
        # 训练单个专家
        experts_to_train = [e.strip() for e in args.expert.split(',')]
        results = train_all_experts(base_config, args.num_epochs, experts=experts_to_train,
                                   save_embeddings=args.save_embeddings,
                                   embeddings_dir=args.embeddings_dir)

    print("\n✓ 所有训练任务完成!")


if __name__ == '__main__':
    main()
