"""
统一专家训练入口
使用通用训练器训练所有专家模型
"""
import torch
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.expert_trainer import ExpertTrainer
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
    # 默认训练的专家列表
    if experts is None:
        experts = ['des', 'tweets']  # 后续可添加 'graph' 等

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
                        help='要训练的专家 (des, tweets, all)')
    parser.add_argument('--dataset_path', type=str, default='./processed_data',
                        help='数据集路径')
    parser.add_argument('--checkpoint_dir', type=str, default='../autodl-tmp/checkpoints',
                        help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率 (None表示使用默认值)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda 或 cpu)')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='BERT模型名称 (用于Description Expert)')
    parser.add_argument('--roberta_model', type=str, default='distilroberta-base',
                        help='RoBERTa模型名称 (用于Tweets Expert)')
    parser.add_argument('--freeze_bert', action='store_true', default=True,
                        help='是否冻结BERT参数 (仅训练MLP)')

    args = parser.parse_args()

    # 确定设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"\n使用设备: {device}")

    # 基础配置参数
    base_config = {
        'dataset_path': args.dataset_path,
        'batch_size': args.batch_size,
        'device': device,
        'checkpoint_dir': args.checkpoint_dir,
        'bert_model_name': args.bert_model,
        'roberta_model_name': args.roberta_model,
        'freeze_bert': args.freeze_bert,
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
    # 如果不使用命令行参数，可以直接调用
    # main()

    # 或者直接配置参数运行
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_params = {
        'dataset_path': './processed_data',
        'batch_size': 32,
        'device': device,
        'checkpoint_dir': '../autodl-tmp/checkpoints',
        'bert_model_name': 'bert-base-uncased',
        'roberta_model_name': 'distilroberta-base',
        'freeze_bert': True,
    }

    # 训练所有专家
    results = train_all_experts(config_params, num_epochs=10, experts=['des', 'tweets'])

    # 或者只训练单个专家
    # results = train_single_expert('des', config_params, num_epochs=10)

