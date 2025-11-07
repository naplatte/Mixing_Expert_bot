from dataset import Twibot20
from model import DesExpert
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数设置
dropout = 0.3
lr = 1e-4  # 减小学习率以防止过拟合
weight_decay = 5e-4
epochs = 50
batch_size = 64
early_stopping_patience = 8

# 数据加载
print("加载数据集...")
dataset = Twibot20(device=device, process=True, save=True)
des_tensor, train_idx, val_idx, test_idx = dataset.dataloader()
labels = dataset.load_labels()

# 准备数据集
train_data = TensorDataset(des_tensor[list(train_idx)], labels[list(train_idx)])
val_data = TensorDataset(des_tensor[list(val_idx)], labels[list(val_idx)])
test_data = TensorDataset(des_tensor[list(test_idx)], labels[list(test_idx)])

# 数据加载器
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 初始化模型
model = DesExpert(input_dim=768, expert_dim=64, dropout_rate=dropout).to(device)

# 定义损失函数（二元交叉熵损失）
criterion = nn.BCELoss()

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 训练函数
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        # 前向传播
        _, bot_probs = model(inputs)
        
        # 计算损失
        loss = criterion(bot_probs, targets)
        
        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        # 收集预测结果和真实标签
        preds = (bot_probs >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1, mcc

# 评估函数
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_expert_vectors = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
            
            # 前向传播
            expert_vectors, bot_probs = model(inputs)
            
            # 计算损失
            loss = criterion(bot_probs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
            # 收集结果
            preds = (bot_probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_expert_vectors.extend(expert_vectors.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1, mcc, np.array(all_expert_vectors)

# 训练循环
best_val_loss = float('inf')
best_model_state = None
patience_counter = 0
train_losses = []
val_losses = []
train_f1s = []
val_f1s = []

print("开始训练模型...")
start_time = time.time()

for epoch in range(epochs):
    # 训练
    train_loss, train_acc, train_prec, train_rec, train_f1, train_mcc = train_epoch(model, train_loader, criterion, optimizer)
    
    # 验证
    val_loss, val_acc, val_prec, val_rec, val_f1, val_mcc, _ = evaluate(model, val_loader, criterion)
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # 记录指标
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)
    
    # 打印当前轮次的结果
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, MCC: {train_mcc:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, MCC: {val_mcc:.4f}")
    
    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"  保存最佳模型...")
    else:
        patience_counter += 1
        print(f"  早停计数: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print(f"  早停触发，停止训练...")
            break

training_time = time.time() - start_time
print(f"训练完成，耗时: {training_time:.2f}秒")

# 加载最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("加载最佳模型状态")

# 在测试集上评估
print("在测试集上评估模型...")
test_loss, test_acc, test_prec, test_rec, test_f1, test_mcc, test_expert_vectors = evaluate(model, test_loader, criterion)

print("\n测试结果:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"MCC: {test_mcc:.4f}")

# 保存模型
model_save_path = './best_model.pt'
torch.save(model.state_dict(), model_save_path)
print(f"\n模型已保存至: {model_save_path}")

# 保存专家向量
np.save('./test_expert_vectors.npy', test_expert_vectors)
print("测试集的专家向量已保存至: test_expert_vectors.npy")

# 绘制训练曲线
try:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label='Training F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.title('F1 Score vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./training_curves.png')
    print("训练曲线已保存至: training_curves.png")
except Exception as e:
    print(f"绘制训练曲线时出错: {e}")

print("\n任务完成！")

