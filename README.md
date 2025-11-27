# 混合专家社交机器人检测

基于混合专家系统（Mixture of Experts）的社交机器人检测项目。

## 项目结构

```
├── expert_trainer.py          # 通用专家训练器
├── expert_configs.py          # 专家配置（数据集、模型）
├── train_experts.py           # 统一训练入口
├── model.py                   # 专家模型定义
├── dataset.py                 # 数据集处理
├── metrics.py                 # 评估指标（F1、Accuracy等）
└── train_gating_network.py    # 门控网络训练
```

## 快速开始

### 训练所有专家
```bash
python train_experts.py
```

### 训练单个专家
```bash
# Description 专家
python train_experts.py --expert des

# Tweets 专家
python train_experts.py --expert tweets
```

### 自定义参数
```bash
python train_experts.py --expert des,tweets --num_epochs 15 --batch_size 64
```

### 查看所有参数
```bash
python train_experts.py --help
```

## 添加新专家（3步）

### 1. 在 expert_configs.py 添加配置函数
```python
def create_your_expert_config(
    dataset_path='./processed_data',
    batch_size=32,
    learning_rate=1e-3,
    device='cuda',
    checkpoint_dir='./checkpoints'
):
    # 1. 加载数据
    dataset = load_your_data()
    
    # 2. 创建数据加载器
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)
    
    # 3. 初始化模型
    model = YourExpertModel().to(device)
    
    # 4. 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # 5. 定义数据提取函数
    def extract_fn(batch, device):
        # 将batch数据转换为模型输入格式
        inputs = (batch['data'].to(device),)
        labels = batch['label'].to(device).unsqueeze(1)
        return inputs, labels
    
    return {
        'name': 'your_expert',
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'checkpoint_dir': checkpoint_dir,
        'extract_fn': extract_fn
    }

# 注册到配置表
EXPERT_CONFIGS['your_expert'] = create_your_expert_config
```

### 2. 运行训练
```bash
python train_experts.py --expert your_expert
```

## 已实现的专家

### Description Expert
- 输入：用户简介文本
- 模型：BERT (冻结) + MLP
- 输出：64维专家表示 + Bot概率

### Tweets Expert  
- 输入：用户推文列表
- 模型：RoBERTa (冻结) + MLP
- 输出：64维专家表示 + Bot概率

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--expert` | 要训练的专家 (des, tweets, all) | `all` |
| `--num_epochs` | 训练轮数 | `10` |
| `--batch_size` | 批次大小 | `32` |
| `--learning_rate` | 学习率 | 自动设置 |
| `--device` | 设备 (cuda/cpu) | 自动检测 |
| `--checkpoint_dir` | 模型保存目录 | `../autodl-tmp/checkpoints` |
| `--dataset_path` | 数据集路径 | `./processed_data` |

## 输出文件

训练后会生成：
- `{expert}_expert_best.pt` - 最佳模型（验证集loss最低）
- `{expert}_expert_final.pt` - 最终模型
- `{expert}_expert_history.json` - 训练历史记录

## 数据集

Twibot-20 数据集：
- 训练集：8,278 样本
- 验证集：2,365 样本  
- 测试集：1,183 样本


