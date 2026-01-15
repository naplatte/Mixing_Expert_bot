# Graph专家训练优化说明

## 修改内容

针对Graph专家的训练流程进行了优化，取消了每轮验证时保存最优模型的逻辑。

## 修改位置

文件：`scripts/expert_trainer.py`

### 1. `validate()` 方法
- **修改前**：每轮验证后，如果loss降低则保存最优模型
- **修改后**：Graph专家跳过保存逻辑，仅记录指标

```python
# Graph专家不在训练过程中保存最优模型，只保存最终模型
if self.name == 'graph':
    return metrics
```

### 2. `test()` 方法
- **修改前**：测试前加载best模型
- **修改后**：Graph专家直接使用最后一轮的模型

```python
# Graph专家不加载best模型，直接使用最后一轮的模型
if self.name != 'graph':
    self._load_checkpoint('best')
```

### 3. `extract_and_save_embeddings()` 方法
- **修改前**：提取嵌入前加载best模型
- **修改后**：Graph专家直接使用当前模型

```python
# Graph专家不加载best模型，直接使用最后一轮的模型
if self.name != 'graph':
    self._load_checkpoint('best')
```

## 效果

1. **减少I/O开销**：Graph专家训练时不再每轮保存检查点
2. **保留final模型**：训练结束后仍会保存`graph_expert_final.pt`
3. **其他专家不受影响**：Cat/Num/Des/Post专家仍然保存每轮最优模型

## 使用方式

训练命令不变：
```bash
python scripts/train_experts.py --expert graph --num_epochs 30
```

最终保存的模型：`graph_expert_final.pt`

