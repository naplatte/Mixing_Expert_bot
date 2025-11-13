import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
from dataset import Twibot20
from transformers import BertTokenizer
from model import DesExpert, TweetsExpert, ExpertGatedAggregator
from metrics import update_binary_counts, compute_binary_f1

# 联合数据集：同时提供 des 文本（用于 BERT 分词）、tweets 文本列表、标签，以及专家激活掩码
class CombinedDataset(Dataset):
    def __init__(self, descriptions, tweets_list, labels, tokenizer, max_length=128):
        self.descriptions = descriptions
        self.tweets_list = tweets_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        # 取出该样本的简介文本与标签
        desc = str(self.descriptions[idx])
        label = self.labels[idx]

        # 对简介文本进行 BERT 分词，供 DesExpert 使用
        encoded = self.tokenizer(
            desc,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 取出该样本的 tweets 列表，并清理空与 'None'
        user_tweets = self.tweets_list[idx]
        cleaned_tweets = []
        for t in user_tweets:
            s = str(t).strip()
            if s != '' and s != 'None':
                cleaned_tweets.append(s)
        # 若无有效推文，保留一个空字符串以触发 TweetsExpert 的零表示逻辑
        if len(cleaned_tweets) == 0:
            cleaned_tweets = ['']

        # 专家激活标记：des 有文本且不为 'None'；tweets 至少有一条非空推文
        des_active = 1.0 if desc != 'None' else 0.0
        tw_active = 1.0 if not (len(cleaned_tweets) == 1 and cleaned_tweets[0] == '') else 0.0

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'tweets_text': cleaned_tweets,
            'label': torch.tensor(label, dtype=torch.float32),
            'active_mask': torch.tensor([des_active, tw_active], dtype=torch.float32)
        }

# 组装 batch：
# - des 输入拼成张量（input_ids/attention_mask）
# - tweets 保持为列表（每个样本是变长推文列表）
# - active_mask 为 [batch, num_experts] 的张量
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch]).unsqueeze(1)
    tweets_text_list = [item['tweets_text'] for item in batch]
    active_mask = torch.stack([item['active_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'tweets_text_list': tweets_text_list,
        'active_mask': active_mask
    }

# 加载并冻结 DesExpert（只推理，不训练）
def load_and_freeze_des(des_ckpt_path, bert_model_name, device):
    des = DesExpert(bert_model_name=bert_model_name).to(device)
    ckpt = torch.load(des_ckpt_path, map_location=device)
    des.load_state_dict(ckpt['model_state_dict'])
    des.eval()
    for p in des.parameters():
        p.requires_grad = False
    return des

# 加载并冻结 TweetsExpert（只推理，不训练）
def load_and_freeze_tweets(tw_ckpt_path, device):
    tw = TweetsExpert(device=device).to(device)
    ckpt = torch.load(tw_ckpt_path, map_location=device)
    tw.load_state_dict(ckpt['model_state_dict'])
    tw.eval()
    for p in tw.parameters():
        p.requires_grad = False
    return tw

# 训练门控网络：
# - 冻结最优专家模型，按训练/验证/测试划分迭代
# - 门控接收拼接后的专家表示，输出权重，最终概率按加权和计算
def train_gating_network(
    dataset_path='./processed_data',
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='../autodl-tmp/checkpoints',
    bert_model_name='bert-base-uncased',
    des_ckpt_path='../autodl-tmp/checkpoints/des_expert_best.pt',
    tw_ckpt_path='../autodl-tmp/checkpoints/tweets_expert_best.pt'
):
    os.makedirs(save_dir, exist_ok=True)
    # 加载数据与原始文本
    dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    descriptions = dataset.Des_preprocess()
    tweets_list = dataset.tweets_preprogress()
    labels = dataset.load_labels().cpu().numpy()
    if isinstance(descriptions, np.ndarray):
        descriptions = descriptions.tolist()
    if isinstance(tweets_list, np.ndarray):
        tweets_list = tweets_list.tolist()
    train_idx, val_idx, test_idx = dataset.train_val_test_mask()
    train_idx, val_idx, test_idx = list(train_idx), list(val_idx), list(test_idx)
    train_desc = [descriptions[i] for i in train_idx]
    val_desc = [descriptions[i] for i in val_idx]
    test_desc = [descriptions[i] for i in test_idx]
    train_tw = [tweets_list[i] for i in train_idx]
    val_tw = [tweets_list[i] for i in val_idx]
    test_tw = [tweets_list[i] for i in test_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    # 构建联合数据集与 DataLoader
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_ds = CombinedDataset(train_desc, train_tw, train_labels, tokenizer)
    val_ds = CombinedDataset(val_desc, val_tw, val_labels, tokenizer)
    test_ds = CombinedDataset(test_desc, test_tw, test_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # 加载并冻结两个专家；初始化门控聚合器
    des_expert = load_and_freeze_des(des_ckpt_path, bert_model_name, device)
    tweets_expert = load_and_freeze_tweets(tw_ckpt_path, device)
    aggregator = ExpertGatedAggregator(num_experts=2, expert_dim=64, hidden_dim=128).to(device)
    criterion = nn.BCELoss()
    optimizer = AdamW(aggregator.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        aggregator.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_counts = {'tp': 0, 'fp': 0, 'fn': 0}
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_t = batch['label'].to(device)
            tweets_text_list = batch['tweets_text_list']
            active_mask = batch['active_mask'].to(device)
            # 冻结专家，前向只用于生成 64d 表示与概率；门控参与反向传播
            optimizer.zero_grad()
            with torch.no_grad():
                des_repr, des_prob = des_expert(input_ids, attention_mask)
                tw_repr, tw_prob = tweets_expert(tweets_text_list)
            final_prob, _ = aggregator([des_repr, tw_repr], [des_prob, tw_prob], active_mask)
            loss = criterion(final_prob, labels_t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (final_prob > 0.5).float()
            train_correct += (preds == labels_t).sum().item()
            train_total += labels_t.size(0)
            update_binary_counts(preds, labels_t, train_counts)
            _, _, f1_running = compute_binary_f1(train_counts)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}', 'f1': f'{f1_running:.4f}'})
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        _, _, train_f1 = compute_binary_f1(train_counts)
        aggregator.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_counts = {'tp': 0, 'fp': 0, 'fn': 0}
        # 验证阶段：评估门控在验证集的表现，选择最优
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_t = batch['label'].to(device)
                tweets_text_list = batch['tweets_text_list']
                active_mask = batch['active_mask'].to(device)
                des_repr, des_prob = des_expert(input_ids, attention_mask)
                tw_repr, tw_prob = tweets_expert(tweets_text_list)
                final_prob, _ = aggregator([des_repr, tw_repr], [des_prob, tw_prob], active_mask)
                loss = criterion(final_prob, labels_t)
                val_loss += loss.item()
                preds = (final_prob > 0.5).float()
                val_correct += (preds == labels_t).sum().item()
                val_total += labels_t.size(0)
                update_binary_counts(preds, labels_t, val_counts)
                _, _, f1_running = compute_binary_f1(val_counts)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{val_correct/val_total:.4f}', 'f1': f'{f1_running:.4f}'})
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        _, _, val_f1 = compute_binary_f1(val_counts)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'gating_best.pt')
            torch.save({'epoch': epoch, 'model_state_dict': aggregator.state_dict(), 'val_loss': avg_val_loss, 'val_acc': val_acc, 'val_f1': val_f1}, model_path)
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val   F1: {val_f1:.4f}")
    print("\nTesting best gating...")
    # 使用验证集最优的门控权重进行测试评估
    best = torch.load(os.path.join(save_dir, 'gating_best.pt'), map_location=device)
    aggregator.load_state_dict(best['model_state_dict'])
    aggregator.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_counts = {'tp': 0, 'fp': 0, 'fn': 0}
    # 测试阶段：冻结专家 + 最优门控，计算最终指标
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_t = batch['label'].to(device)
            tweets_text_list = batch['tweets_text_list']
            active_mask = batch['active_mask'].to(device)
            des_repr, des_prob = des_expert(input_ids, attention_mask)
            tw_repr, tw_prob = tweets_expert(tweets_text_list)
            final_prob, _ = aggregator([des_repr, tw_repr], [des_prob, tw_prob], active_mask)
            loss = criterion(final_prob, labels_t)
            test_loss += loss.item()
            preds = (final_prob > 0.5).float()
            test_correct += (preds == labels_t).sum().item()
            test_total += labels_t.size(0)
            update_binary_counts(preds, labels_t, test_counts)
    avg_test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total
    _, _, test_f1 = compute_binary_f1(test_counts)
    final_path = os.path.join(save_dir, 'gating_final.pt')
    torch.save({'model_state_dict': aggregator.state_dict(), 'test_loss': avg_test_loss, 'test_acc': test_acc, 'test_f1': test_f1}, final_path)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Test F1:   {test_f1:.4f}")
    print(f"Saved final gating to: {final_path}")
    return aggregator

if __name__ == '__main__':
    config = {
        'dataset_path': './processed_data',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '../autodl-tmp/checkpoints',
        'bert_model_name': 'bert-base-uncased',
        'des_ckpt_path': '../autodl-tmp/checkpoints/des_expert_best.pt',
        'tw_ckpt_path': '../autodl-tmp/checkpoints/tweets_expert_best.pt'
    }
    print(f"Using device: {config['device']}")
    aggregator = train_gating_network(**config)
