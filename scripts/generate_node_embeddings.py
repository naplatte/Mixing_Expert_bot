"""生成所有节点的Cat/Num嵌入供GraphExpert使用"""
import torch
import argparse
import sys
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import Twibot20
from src.model import CatExpert, NumExpert


def generate_embeddings(device='cuda', checkpoint_dir='../../autodl-fs/model',
                        output_dir='../../autodl-fs/node_embedding'):
    """使用训练好的Cat/Num专家生成所有节点的嵌入"""
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # 加载全量数据
    print("\n加载全量数据...")
    twibot = Twibot20(device=device, process=True, save=True)
    num_all_nodes = len(twibot.df_data)
    print(f"总节点数: {num_all_nodes}")

    # 处理Cat特征
    print("\n" + "="*60)
    print("生成 Category 嵌入")
    print("="*60)
    cat_features_all = twibot.cat_prop_preprocess(all_nodes=True)
    print(f"Cat特征维度: {cat_features_all.shape}")

    cat_model = CatExpert(input_dim=3, device=device).to(device)
    cat_ckpt_path = os.path.join(checkpoint_dir, 'cat_expert_best.pt')
    if os.path.exists(cat_ckpt_path):
        ckpt = torch.load(cat_ckpt_path, map_location=device)
        cat_model.load_state_dict(ckpt['model_state_dict'])
        print(f"加载Cat检查点: {cat_ckpt_path}")
    else:
        print(f"警告: Cat检查点不存在，使用随机初始化")

    cat_model.eval()
    with torch.no_grad():
        batch_size = 4096
        cat_embeddings = []
        for i in range(0, num_all_nodes, batch_size):
            batch = cat_features_all[i:i+batch_size]
            emb, _ = cat_model(batch)
            cat_embeddings.append(emb.cpu())
        cat_embeddings = torch.cat(cat_embeddings, dim=0)

    cat_save_path = os.path.join(output_dir, 'node_cat_embeddings.pt')
    torch.save(cat_embeddings, cat_save_path)
    print(f"Cat嵌入已保存: {cat_save_path}, shape={cat_embeddings.shape}")

    # 处理Num特征
    print("\n" + "="*60)
    print("生成 Numerical 嵌入")
    print("="*60)
    num_features_all = twibot.num_prop_preprocess(all_nodes=True)
    print(f"Num特征维度: {num_features_all.shape}")

    num_model = NumExpert(input_dim=5, device=device).to(device)
    num_ckpt_path = os.path.join(checkpoint_dir, 'num_expert_best.pt')
    if os.path.exists(num_ckpt_path):
        ckpt = torch.load(num_ckpt_path, map_location=device)
        num_model.load_state_dict(ckpt['model_state_dict'])
        print(f"加载Num检查点: {num_ckpt_path}")
    else:
        print(f"警告: Num检查点不存在，使用随机初始化")

    num_model.eval()
    with torch.no_grad():
        num_embeddings = []
        for i in range(0, num_all_nodes, batch_size):
            batch = num_features_all[i:i+batch_size]
            emb, _ = num_model(batch)
            num_embeddings.append(emb.cpu())
        num_embeddings = torch.cat(num_embeddings, dim=0)

    num_save_path = os.path.join(output_dir, 'node_num_embeddings.pt')
    torch.save(num_embeddings, num_save_path)
    print(f"Num嵌入已保存: {num_save_path}, shape={num_embeddings.shape}")

    print("\n" + "="*60)
    print("嵌入生成完成!")
    print("="*60)
    print(f"  Cat: {cat_embeddings.shape}")
    print(f"  Num: {num_embeddings.shape}")

    return {'cat': cat_embeddings, 'num': num_embeddings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成节点嵌入')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/model')
    parser.add_argument('--output_dir', type=str, default='../../autodl-fs/node_embedding')
    args = parser.parse_args()

    generate_embeddings(args.device, args.checkpoint_dir, args.output_dir)

