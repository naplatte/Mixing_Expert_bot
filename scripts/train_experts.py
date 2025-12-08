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
def train_single_expert(expert_name, config_params, num_epochs=10):
    # 获取专家配置
    config = get_expert_config(expert_name, **config_params)

    # 创建训练器
    trainer = ExpertTrainer(config)

    # 开始训练
    history = trainer.train(num_epochs)

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
        'graph': ['des', 'tweets'],  # 图专家依赖 des 和 tweets
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


def train_all_experts(config_params, num_epochs=10, experts=None):
    """
    训练所有专家或指定的专家列表

    Args:
        config_params: 配置参数字典
        num_epochs: 训练轮数
        experts: 要训练的专家列表，None表示训练所有可用专家

    Returns:
        dict: 所有专家的训练结果
    """
    # 默认训练的专家列表（按依赖顺序）
    if experts is None:
        experts = ['des', 'tweets', 'graph']  # graph 依赖 des 和 tweets，所以放最后

    results = {}

    print("\n" + "="*60)
    print("开始训练所有专家模型")
    print("="*60)
    print(f"专家列表: {experts}")
    print(f"训练轮数: {num_epochs}")
    print(f"设备: {config_params['device']}")
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
            history = train_single_expert(expert_name, config_params, num_epochs)
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
                        help='要训练的专家 (des, tweets, graph, all)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='数据集路径')
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/checkpoints',
                        help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率 (None表示使用默认值)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda 或 cpu)')

    # des和tweets Expert 参数
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='BERT模型名称 (用于Description Expert)')
    parser.add_argument('--roberta_model', type=str, default='distilroberta-base',
                        help='RoBERTa模型名称 (用于Tweets Expert)')
    parser.add_argument('--freeze_bert', action='store_true', default=True,
                        help='是否冻结BERT参数 (仅训练MLP)')

    # Graph Expert 参数
    parser.add_argument('--graph_hidden_dim', type=int, default=128,
                        help='图专家隐藏层维度')
    parser.add_argument('--graph_num_layers', type=int, default=2,
                        help='图专家RGCN层数')
    parser.add_argument('--graph_dropout', type=float, default=0.3,
                        help='图专家Dropout比率')
    parser.add_argument('--graph_expert_names', type=str, default='des,tweets',
                        help='图专家依赖的专家列表（逗号分隔）')

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
        # Text Expert 参数
        'bert_model_name': args.bert_model,
        'roberta_model_name': args.roberta_model,
        'freeze_bert': args.freeze_bert,
        # Graph Expert 参数
        'hidden_dim': args.graph_hidden_dim,
        'num_layers': args.graph_num_layers,
        'dropout': args.graph_dropout,
        'expert_names': [e.strip() for e in args.graph_expert_names.split(',')],
    }

    # 添加学习率 (如果指定)
    if args.learning_rate is not None:
        base_config['learning_rate'] = args.learning_rate

    # 训练专家
    if args.expert == 'all':
        # 训练所有专家
        results = train_all_experts(base_config, args.num_epochs)
    else:
        # 训练单个专家
        experts_to_train = [e.strip() for e in args.expert.split(',')]
        results = train_all_experts(base_config, args.num_epochs, experts=experts_to_train)

    print("\n✓ 所有训练任务完成!")


if __name__ == '__main__':
    main()
