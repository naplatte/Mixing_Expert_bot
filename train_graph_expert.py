import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
from dataset import Twibot20
from model import GraphExpert
from metrics import update_binary_counts, compute_binary_f1


class GraphDataset(Dataset):
    """
    图结构数据集
    每个样本包含：节点索引（在完整图中的索引）和标签
    """
    def __init__(self, node_indices, labels):
        """
        Args:
            node_indices: 节点索引列表（在完整图 df_data 中的索引）
            labels: 对应的标签列表
        """
        self.node_indices = node_indices
        self.labels = labels
    
    def __len__(self):
        return len(self.node_indices)
    
    def __getitem__(self, idx):
        return {
            'node_idx': self.node_indices[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def collate_fn(batch):
    """组装 batch"""
    node_indices = torch.tensor([item['node_idx'] for item in batch], dtype=torch.long)
    labels = torch.stack([item['label'] for item in batch]).unsqueeze(1)
    return {
        'node_indices': node_indices,
        'label': labels
    }


def train_graph_expert(
    dataset_path='./processed_data',
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='../autodl-tmp/checkpoints',
    hidden_dim=128,
    expert_dim=64,
    num_layers=2,
    dropout=0.1
):
    """
    训练 GraphExpert 模型
    
    Args:
        dataset_path: 数据集路径
        batch_size: 批次大小
        learning_rate: 学习率
        num_epochs: 训练轮数
        device: 设备
        save_dir: 模型保存目录
        hidden_dim: 隐藏层维度
        expert_dim: 专家表示维度
        num_layers: RGCN 层数
        dropout: Dropout 率
    """
    print("=" * 60)
    print("开始训练 GraphExpert 模型")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("\n1. 加载数据和构建图...")
    twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    
    # 构建图结构（使用完整数据集，包括 support）
    edge_index, edge_type = twibot_dataset.Build_Graph()
    num_nodes = len(twibot_dataset.df_data)  # 完整图的节点数（包括 support）
    num_labeled_nodes = len(twibot_dataset.df_data_labeled)  # 有标签的节点数
    
    print(f"   图节点总数: {num_nodes}")
    print(f"   有标签节点数: {num_labeled_nodes}")
    print(f"   边数: {edge_index.shape[1]}")
    print(f"   关系类型数: {edge_type.max().item() + 1}")
    
    # 节点结构特征
    node_features = twibot_dataset.get_node_features()
    
    # 获取标签（只对有标签的节点）
    labels = twibot_dataset.load_labels()
    labels = labels.cpu().numpy()
    
    # 获取训练/验证/测试集索引（这些索引是在 df_data_labeled 中的索引）
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)
    
    # 注意：train_idx, val_idx, test_idx 是在 df_data_labeled 中的索引
    # 由于 df_data 的前 num_labeled_nodes 个节点与 df_data_labeled 对应
    # 所以这些索引可以直接用于 df_data（完整图）
    train_node_indices = train_idx
    train_labels = labels[train_idx]
    
    val_node_indices = val_idx
    val_labels = labels[val_idx]
    
    test_node_indices = test_idx
    test_labels = labels[test_idx]
    
    print(f"\n2. 数据集划分:")
    print(f"   训练集: {len(train_node_indices)} 样本")
    print(f"   验证集: {len(val_node_indices)} 样本")
    print(f"   测试集: {len(test_node_indices)} 样本")
    
    # 构建数据集和 DataLoader
    train_ds = GraphDataset(train_node_indices, train_labels)
    val_ds = GraphDataset(val_node_indices, val_labels)
    test_ds = GraphDataset(test_node_indices, test_labels)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 初始化模型
    print("\n3. 初始化模型...")
    model = GraphExpert(
        num_nodes=num_nodes,
        node_features=node_features,
        num_relations=2,  # following 和 follower
        hidden_dim=hidden_dim,
        expert_dim=expert_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    print("\n4. 开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            node_indices = batch['node_indices'].to(device)
            labels_t = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            expert_repr, bot_prob = model(node_indices, edge_index, edge_type)
            loss = criterion(bot_prob, labels_t)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            preds = (bot_prob > 0.5).float()
            train_correct += (preds == labels_t).sum().item()
            train_total += labels_t.size(0)
            update_binary_counts(preds, labels_t, train_counts)
            
            _, _, f1_running = compute_binary_f1(train_counts)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}',
                'f1': f'{f1_running:.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        _, _, train_f1 = compute_binary_f1(train_counts)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in pbar:
                node_indices = batch['node_indices'].to(device)
                labels_t = batch['label'].to(device)
                
                expert_repr, bot_prob = model(node_indices, edge_index, edge_type)
                loss = criterion(bot_prob, labels_t)
                
                val_loss += loss.item()
                preds = (bot_prob > 0.5).float()
                val_correct += (preds == labels_t).sum().item()
                val_total += labels_t.size(0)
                update_binary_counts(preds, labels_t, val_counts)
                
                _, _, f1_running = compute_binary_f1(val_counts)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{val_correct/val_total:.4f}',
                    'f1': f'{f1_running:.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        _, _, val_f1 = compute_binary_f1(val_counts)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'graph_expert_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1
            }, model_path)
            print(f"  ✓ 保存最佳模型 (Val Loss: {avg_val_loss:.4f})")
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val   F1: {val_f1:.4f}")
    
    # 测试
    print("\n5. 测试最佳模型...")
    best = torch.load(os.path.join(save_dir, 'graph_expert_best.pt'), map_location=device)
    model.load_state_dict(best['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_counts = {'tp': 0, 'fp': 0, 'fn': 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            node_indices = batch['node_indices'].to(device)
            labels_t = batch['label'].to(device)
            
            expert_repr, bot_prob = model(node_indices, edge_index, edge_type)
            loss = criterion(bot_prob, labels_t)
            
            test_loss += loss.item()
            preds = (bot_prob > 0.5).float()
            test_correct += (preds == labels_t).sum().item()
            test_total += labels_t.size(0)
            update_binary_counts(preds, labels_t, test_counts)
    
    avg_test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total
    _, _, test_f1 = compute_binary_f1(test_counts)
    
    final_path = os.path.join(save_dir, 'graph_expert_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_loss': avg_test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1
    }, final_path)
    
    print(f"\n测试结果:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.4f}")
    print(f"  Test F1:   {test_f1:.4f}")
    print(f"  模型已保存到: {final_path}")
    
    return model


if __name__ == '__main__':
    config = {
        'dataset_path': './processed_data',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '../autodl-tmp/checkpoints',
        'hidden_dim': 128,
        'expert_dim': 64,
        'num_layers': 2,
        'dropout': 0.1
    }
    
    print(f"使用设备: {config['device']}")
    model = train_graph_expert(**config)

