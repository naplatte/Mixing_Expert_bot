# 专家门控聚合中的数据缺失处理说明

## 问题描述

在混合专家系统中，某些用户可能缺少特定类型的数据（例如没有简介、没有推文等）。之前的实现使用填充值（如空字符串 `['']`）来处理缺失数据，但这会导致该专家仍然参与门控聚合，可能产生不准确的预测。

## 解决方案

通过 **专家激活掩码（Active Mask）** 机制，在门控聚合时动态地排除不可用的专家。

### 核心思想

1. **数据标记**: 在数据集中标记每个样本是否有有效数据
2. **掩码传递**: 将标记传递到门控网络
3. **权重调整**: 门控网络根据掩码将不可用专家的权重设为0，并重新归一化

## 实现细节

### 1. 数据集层面（configs/expert_configs.py）

#### Description Expert

```python
class DescriptionDataset(Dataset):
    def __getitem__(self, idx):
        description = str(self.descriptions[idx])
        
        # 判断是否有有效简介（非空且非'None'）
        has_description = (description.strip() != '' and 
                          description.strip().lower() != 'none')
        
        return {
            'input_ids': ...,
            'attention_mask': ...,
            'label': ...,
            'has_description': has_description  # ✅ 关键标记
        }
```

#### Tweets Expert

```python
class TweetsDataset(Dataset):
    def __getitem__(self, idx):
        user_tweets = self.tweets_list[idx]
        
        # 清理推文文本
        cleaned_tweets = []
        for tweet in user_tweets:
            tweet_str = str(tweet).strip()
            if tweet_str != '' and tweet_str != 'None':
                cleaned_tweets.append(tweet_str)
        
        # 判断是否有有效推文
        has_tweets = len(cleaned_tweets) > 0  # ✅ 关键判断
        
        # 如果没有有效推文，填充空字符串（但标记为不可用）
        if len(cleaned_tweets) == 0:
            cleaned_tweets = ['']
        
        return {
            'tweets_text': cleaned_tweets,
            'label': ...,
            'has_tweets': has_tweets  # ✅ 关键标记
        }

def collate_tweets_fn(batch):
    return {
        'tweets_text_list': ...,
        'label': ...,
        'has_tweets': torch.tensor([item['has_tweets'] for item in batch])  # ✅ 传递标记
    }
```

### 2. 门控网络训练（scripts/train_gating_network.py）

#### 数据集生成激活掩码

```python
class CombinedDataset(Dataset):
    def __getitem__(self, idx):
        # 处理简介
        desc = str(self.descriptions[idx])
        des_active = 1.0 if (desc.strip() != '' and 
                            desc.strip().lower() != 'none') else 0.0
        
        # 处理推文
        cleaned_tweets = [...]  # 清理推文
        has_tweets = len(cleaned_tweets) > 0
        tw_active = 1.0 if has_tweets else 0.0
        
        # 图结构总是可用
        graph_active = 1.0
        
        return {
            'input_ids': ...,
            'tweets_text': ...,
            'label': ...,
            'active_mask': torch.tensor([des_active, tw_active, graph_active])  # ✅ 激活掩码
        }
```

### 3. 门控网络模型（src/model.py）

```python
class ExpertGatedAggregator(nn.Module):
    def forward(self, expert_reprs, expert_probs, active_mask):
        """
        Args:
            expert_reprs: List of [batch_size, expert_dim]
            expert_probs: List of [batch_size, 1]
            active_mask: [batch_size, num_experts]  # ✅ 关键输入
                - 1表示专家可用，0表示不可用
        """
        # 计算原始门控权重
        gate_weights = self.gate_network(gate_input)  # [batch_size, num_experts]
        
        # 应用激活掩码：不可用的专家权重设为0
        gate_weights = gate_weights * active_mask  # ✅ 关键步骤
        
        # 重新归一化（只在可用专家间分配权重）
        gate_weights_sum = gate_weights.sum(dim=1, keepdim=True)
        gate_weights = gate_weights / (gate_weights_sum + 1e-8)  # ✅ 避免除零
        
        # 加权聚合
        weighted_prob = (gate_weights * expert_probs_concat).sum(dim=1, keepdim=True)
        
        return weighted_prob, gate_weights
```

## 工作流程示例

假设有一个用户：
- ✅ 有简介："I am a bot"
- ❌ 没有推文（空）
- ✅ 有图结构

### 数据流

```python
# 1. 数据集生成
has_description = True   # 有简介
has_tweets = False       # 没有推文
active_mask = [1.0, 0.0, 1.0]  # [des, tweets, graph]

# 2. 各专家输出
des_repr = [0.2, 0.5, ...]      # 64维表示
des_prob = 0.8                   # Bot概率

tweets_repr = [0.0, 0.0, ...]   # 零向量（没有推文）
tweets_prob = 0.5                # 填充概率（会被忽略）

graph_repr = [0.3, 0.1, ...]    # 64维表示
graph_prob = 0.6                 # Bot概率

# 3. 门控网络计算权重
raw_weights = [0.4, 0.3, 0.3]           # 原始权重
masked_weights = [0.4, 0.0, 0.3]        # 应用掩码（tweets权重变0）
normalized_weights = [0.571, 0.0, 0.429] # 重新归一化（只在des和graph间分配）

# 4. 最终聚合
final_prob = 0.571 * 0.8 + 0.0 * 0.5 + 0.429 * 0.6
           = 0.457 + 0.0 + 0.257
           = 0.714  # ✅ Tweets专家被成功排除
```

## 关键优势

### 1. 动态专家选择
- 根据数据可用性自动调整专家组合
- 不同样本可以使用不同的专家子集

### 2. 避免噪声
- 缺失数据不会产生无意义的填充表示
- 提高预测准确性

### 3. 可解释性
- `gate_weights` 清楚显示每个专家的贡献
- 权重为0明确表示该专家未参与

## 使用示例

### 训练门控网络

```python
# 1. 加载预训练的专家模型
des_expert = load_and_freeze_des(...)
tweets_expert = load_and_freeze_tweets(...)
graph_expert = load_and_freeze_graph(...)

# 2. 训练循环
for batch in train_loader:
    # 获取激活掩码
    active_mask = batch['active_mask'].to(device)  # [batch_size, 3]
    
    # 各专家前向传播
    des_repr, des_prob = des_expert(batch['input_ids'], batch['attention_mask'])
    tw_repr, tw_prob = tweets_expert(batch['tweets_text_list'])
    gr_repr, gr_prob = graph_expert(batch['node_indices'], edge_index, edge_type)
    
    # 门控聚合（自动处理缺失数据）
    final_prob, gate_weights = gating_network(
        [des_repr, tw_repr, gr_repr],
        [des_prob, tw_prob, gr_prob],
        active_mask  # ✅ 传入激活掩码
    )
```

### 查看专家权重分布

```python
# 查看某个样本的专家权重
sample_weights = gate_weights[0]  # [des, tweets, graph]
print(f"Description: {sample_weights[0]:.3f}")
print(f"Tweets: {sample_weights[1]:.3f}")      # 如果为0，说明该用户没有推文
print(f"Graph: {sample_weights[2]:.3f}")
```

## 注意事项

### 1. 至少一个专家可用
确保每个样本至少有一个专家可用（active_mask不全为0），否则会导致除零错误。在实践中：
- 图结构数据通常总是可用（`graph_active = 1.0`）
- 可以在门控网络中添加保护机制

### 2. 训练数据平衡
如果某个专家很少被激活，可能会影响其训练效果。可以考虑：
- 数据增强
- 调整损失函数权重

### 3. 性能影响
激活掩码的计算和应用对性能影响很小（仅涉及简单的乘法和归一化操作）。

## 总结

通过引入 **专家激活掩码（Active Mask）**，成功实现了：

✅ **数据缺失感知**: 自动识别每个样本的数据可用性  
✅ **动态专家选择**: 只使用有有效数据的专家  
✅ **权重重新分配**: 在可用专家间重新归一化权重  
✅ **提高准确性**: 避免填充值带来的噪声  
✅ **可解释性强**: 权重为0明确表示专家未参与  

这种机制使得混合专家系统能够**优雅地处理不完整数据**，提高模型的鲁棒性和预测准确性。

