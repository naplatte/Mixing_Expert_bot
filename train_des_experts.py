import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import os
from dataset import Twibot20
from model import DesExpert
from metrics import update_binary_counts, compute_binary_f1

class DescriptionDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_length=128):
        """
        Args:
            descriptions: description 文本列表
            labels: 对应的标签列表 (0: human, 1: bot)
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
        """
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # 返回数据集中样本的总数
    def __len__(self):
        return len(self.descriptions)

    # 获取单个样本(原始文本),当对对象使用索引操作时会自动调用
    def __getitem__(self, idx):
        description = str(self.descriptions[idx]) # 将文本信息转为字符串
        label = self.labels[idx]
        
        # Tokenize(BERT分词器)
        encoded = self.tokenizer(
            description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True, # 超过最大长度的序列将被截断
            return_tensors='pt' # pytorch tensor
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def train_des_expert(
    dataset_path='./processed_data',
    batch_size=32,
    learning_rate=2e-5,
    num_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='./ExpertModel',
    bert_model_name='bert-base-uncased'
):
    """
    训练 DesExpert 模型
    
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
    print("开始训练 DesExpert 模型")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("\n1. 加载数据...")
    twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    
    # 获取 description 文本和标签
    descriptions = twibot_dataset.Des_preprocess()  # 获取原始文本并预处理
    labels = twibot_dataset.load_labels()  # 获取标签

    if isinstance(descriptions, np.ndarray):
        descriptions = descriptions.tolist() # 如果 descriptions 是 numpy 数组，则将其转换为列表
    labels = labels.cpu().numpy() # label转为numpy数组 (numpy只能直接处理CPU上的数据，无法直接访问GPU上的数据，所以需要先将lable转移到CPU)
    
    # 获取训练/验证/测试集索引,并转换为列表形式,每个列表包含一系列整数值，表示样本在原始数据集中的索引位置
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)
    
    # 提取des以及对应label
    train_descriptions = [descriptions[i] for i in train_idx]
    train_labels = labels[train_idx]
    
    val_descriptions = [descriptions[i] for i in val_idx]
    val_labels = labels[val_idx]
    
    test_descriptions = [descriptions[i] for i in test_idx]
    test_labels = labels[test_idx]
    
    print(f"   训练集: {len(train_descriptions)} 样本")
    print(f"   验证集: {len(val_descriptions)} 样本")
    print(f"   测试集: {len(test_descriptions)} 样本")
    
    # 创建 tokenizer
    print("\n2. 初始化 tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # 创建数据集和数据加载器
    print("\n3. 创建数据加载器...")
    train_dataset = DescriptionDataset(train_descriptions, train_labels, tokenizer)
    val_dataset = DescriptionDataset(val_descriptions, val_labels, tokenizer)
    test_dataset = DescriptionDataset(test_descriptions, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle:是否打乱顺序
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    print("\n4. 初始化模型...")
    model = DesExpert(bert_model_name=bert_model_name).to(device)
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 训练循环
    print("\n5. 开始训练...")
    best_val_loss = float('inf') # 最下损失，初始为float_max
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0 # 累计当前轮所有batch的损失值(float)
        train_correct = 0 # 累计当前轮模型正确预测的样本数量(int)
        train_total = 0 # 累计当前轮处理样本的总数量(int)
        train_counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]') # 进度条对象
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)  # [batch_size, 1]
            
            # Forward pass
            optimizer.zero_grad()
            expert_repr, bot_prob = model(input_ids, attention_mask)
            
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

            # F1 累积与运行时计算
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
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            val_counts = {'tp': 0, 'fp': 0, 'fn': 0}
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                
                expert_repr, bot_prob = model(input_ids, attention_mask)
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
            model_path = os.path.join(save_dir, 'des_expert_best.pt')
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            expert_repr, bot_prob = model(input_ids, attention_mask)
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
    final_model_path = os.path.join(save_dir, 'des_expert_final.pt')
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
        'batch_size': 32,
        'learning_rate': 2e-5,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints',
        'bert_model_name': 'bert-base-uncased'
    }
    
    print(f"使用设备: {config['device']}")
    print(f"BERT 模型: {config['bert_model_name']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"训练轮数: {config['num_epochs']}")
    
    # 开始训练
    model = train_des_expert(**config)

