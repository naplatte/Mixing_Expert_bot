"""统一专家训练入口"""
import torch
import argparse
import sys
from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from expert_trainer import ExpertTrainer
from configs.expert_configs import get_expert_config


def train_expert(expert_name, config_params, num_epochs=10, save_embeddings=True, embeddings_dir='../../autodl-fs/labeled_embedding'):
    """训练单个专家"""
    config = get_expert_config(expert_name, **config_params)
    trainer = ExpertTrainer(config)
    return trainer.train(num_epochs, save_embeddings=save_embeddings, embeddings_dir=embeddings_dir)


def train_all(config_params, num_epochs=10, experts=None, save_embeddings=True, embeddings_dir='../../autodl-fs/labeled_embedding'):
    """训练多个专家"""
    if experts is None:
        experts = ['des', 'post', 'cat', 'num', 'graph']

    print(f"\n{'='*60}\n训练专家: {experts}\n{'='*60}")

    results = {}
    for name in experts:
        print(f"\n{'#'*60}\n训练: {name.upper()}\n{'#'*60}")
        try:
            results[name] = train_expert(name, config_params, num_epochs, save_embeddings, embeddings_dir)
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

    # 打印结果对比
    print(f"\n{'='*60}\n测试结果对比\n{'='*60}")
    print(f"{'专家':<15} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    for name, hist in results.items():
        t = hist.get('test', {})
        print(f"{name.upper():<15} {t.get('acc',0):<12.4f} {t.get('f1',0):<12.4f} "
              f"{t.get('precision',0):<12.4f} {t.get('recall',0):<12.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='训练专家模型')
    parser.add_argument('--expert', type=str, default='all', help='专家名称 (des/post/cat/num/graph/all)')
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_embeddings', action='store_true', default=True)
    parser.add_argument('--embeddings_dir', type=str, default='../../autodl-fs/labeled_embedding')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    config = {
        'dataset_path': str(project_root / 'processed_data'),
        'batch_size': args.batch_size,
        'device': device,
        'checkpoint_dir': args.checkpoint_dir,
    }
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate

    if args.expert == 'all':
        train_all(config, args.num_epochs, save_embeddings=args.save_embeddings, embeddings_dir=args.embeddings_dir)
    else:
        experts = [e.strip() for e in args.expert.split(',')]
        train_all(config, args.num_epochs, experts=experts, save_embeddings=args.save_embeddings, embeddings_dir=args.embeddings_dir)

    print("\n✓ 完成!")


if __name__ == '__main__':
    main()
