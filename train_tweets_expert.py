"""
训练 TweetsExpert 模型
使用 BERT + MLP 处理推文信息，对用户的多条推文做平均聚合，输出 64维 Expert Representation 和 Bot Probability
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
from dataset import Twibot20
from model import TweetsExpert
from metrics import update_binary_counts, compute_binary_f1

class TweetsDataset(Dataset):
    def __init__(self, tweets_list, labels):
        """
        Args:
            tweets_list: 推文列表，每个元素是一个用户的推文列表（list of strings）
            labels: 对应的标签列表 (0: human, 1: bot)
        """
        self.tweets_list = tweets_list
        self.labels = labels
    
    def __len__(self):
        return len(self.tweets_list)
    
    def __getitem__(self, idx):
        user_tweets = self.tweets_list[idx]  # 该用户的所有推文（list of strings）
        label = self.labels[idx]
        
        # 清理推文文本（去除空和None）
        cleaned_tweets = []
        for tweet in user_tweets:
            tweet_str = str(tweet).strip()
            if tweet_str != '' and tweet_str != 'None':
                cleaned_tweets.append(tweet_str)
        
        # 如果所有推文都是空的，返回空列表
        if len(cleaned_tweets) == 0:
            cleaned_tweets = ['']
        
        return {
            'tweets_text': cleaned_tweets,
            'label': torch.tensor(label, dtype=torch.float32)
        }

# 处理变长的推文列表
def collate_fn(batch):
    tweets_text_lists = [item['tweets_text'] for item in batch] # 1个batch中所有用户的推文列表 eg[["i love u","hello world"],["u love me","haha haha"]]
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'tweets_text_list': tweets_text_lists,
        'label': labels
    }

def train_tweets_expert(
    dataset_path='./processed_data',
    batch_size=32,  # 现在不需要微调BERT，可以使用更大的batch size
    learning_rate=1e-3,  # 只训练MLP，可以使用更大的学习率
    num_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='../autodl-tmp/checkpoints'
):
    """
    训练 TweetsExpert 模型
    
    Args:
        dataset_path: 数据集路径
        batch_size: 批次大小
        learning_rate: 学习率
        num_epochs: 训练轮数
        device: 设备 ('cuda' 或 'cpu')
        save_dir: 模型保存目录
        bert_model_name: BERT 模型名称
    """
    print("=" * 60)
    print("开始训练 TweetsExpert 模型")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("\n1. 加载数据...")
    twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    
    # 获取推文文本和标签
    tweets_list = twibot_dataset.tweets_preprogress()  # 获取推文列表
    labels = twibot_dataset.load_labels()  # 获取标签
    
    # 转换为列表格式
    if isinstance(tweets_list, np.ndarray):
        tweets_list = tweets_list.tolist()
    labels = labels.cpu().numpy()
    
    # 获取训练/验证/测试集索引
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)
    
    # 划分数据集
    train_tweets = [tweets_list[i] for i in train_idx]
    train_labels = labels[train_idx]
    
    val_tweets = [tweets_list[i] for i in val_idx]
    val_labels = labels[val_idx]
    
    test_tweets = [tweets_list[i] for i in test_idx]
    test_labels = labels[test_idx]
    
    print(f"   训练集: {len(train_tweets)} 样本")
    print(f"   验证集: {len(val_tweets)} 样本")
    print(f"   测试集: {len(test_tweets)} 样本")
    
    # 统计推文数量
    train_tweet_counts = [len(tweets) if isinstance(tweets, list) else 0 for tweets in train_tweets]
    print(f"   训练集平均推文数: {np.mean(train_tweet_counts):.2f}")
    print(f"   训练集最大推文数: {np.max(train_tweet_counts)}")
    print(f"   训练集最小推文数: {np.min(train_tweet_counts)}")
    
    # 创建数据集和数据加载器（不再需要tokenizer，直接使用文本）
    print("\n2. 创建数据加载器...")
    train_dataset = TweetsDataset(train_tweets, train_labels)
    val_dataset = TweetsDataset(val_tweets, val_labels)
    test_dataset = TweetsDataset(test_tweets, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 初始化模型
    print("\n3. 初始化模型...")
    model = TweetsExpert(roberta_model_name='distilroberta-base', device=device).to(device)
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 损失函数和优化器
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 训练循环
    print("\n5. 开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            tweets_text_list = batch['tweets_text_list']  # List of lists of strings
            labels = batch['label'].to(device).unsqueeze(1)  # [batch_size, 1]
            
            # Forward pass
            optimizer.zero_grad()
            expert_repr, bot_prob = model(tweets_text_list)
            
            # 计算损失
            loss = criterion(bot_prob, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            predictions = (bot_prob > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            update_binary_counts(predictions, labels, train_counts)
            _, _, f1_running = compute_binary_f1(train_counts)

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}',
                'f1': f'{f1_running:.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        _, _, train_f1 = compute_binary_f1(train_counts)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_pbar:
                tweets_text_list = batch['tweets_text_list']
                labels = batch['label'].to(device).unsqueeze(1)
                
                expert_repr, bot_prob = model(tweets_text_list)
                loss = criterion(bot_prob, labels)
                
                val_loss += loss.item()
                predictions = (bot_prob > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

                update_binary_counts(predictions, labels, val_counts)
                _, _, f1_running = compute_binary_f1(val_counts)

                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{val_correct/val_total:.4f}',
                    'f1': f'{f1_running:.4f}'
                })

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        _, _, val_f1 = compute_binary_f1(val_counts)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'tweets_expert_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, model_path)
            print(f"  ✓ 保存最佳模型到: {model_path}")
    
    # 测试阶段
    print("\n6. 测试模型...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_counts = {'tp': 0, 'fp': 0, 'fn': 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            tweets_text_list = batch['tweets_text_list']
            labels = batch['label'].to(device).unsqueeze(1)
            
            expert_repr, bot_prob = model(tweets_text_list)
            loss = criterion(bot_prob, labels)
            
            test_loss += loss.item()
            predictions = (bot_prob > 0.5).float()
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

            update_binary_counts(predictions, labels, test_counts)
    
    avg_test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total
    _, _, test_f1 = compute_binary_f1(test_counts)
    
    print(f"\n测试结果:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'tweets_expert_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_loss': avg_test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1,
    }, final_model_path)
    print(f"\n✓ 保存最终模型到: {final_model_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    return model

if __name__ == '__main__':
    # 训练参数
    config = {
        'dataset_path': './processed_data',
        'batch_size': 32,  # 现在不需要微调BERT，可以使用更大的batch size
        'learning_rate': 1e-3,  # 只训练MLP，可以使用更大的学习率
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '../autodl-tmp/checkpoints',
    }
    
    print(f"使用设备: {config['device']}")
    print(f"RoBERTa 模型: distilroberta-base (不微调，只用于特征提取)")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"训练轮数: {config['num_epochs']}")
    
    # 开始训练
    model = train_tweets_expert(**config)

