import torch

# 计算'tp', 'fp', 'fn'
def update_binary_counts(predictions: torch.Tensor, labels: torch.Tensor, counts: dict) -> dict:
    pred_int = predictions.long().view(-1)
    true_int = labels.long().view(-1)

    tp = ((pred_int == 1) & (true_int == 1)).sum().item()
    fp = ((pred_int == 1) & (true_int == 0)).sum().item()
    fn = ((pred_int == 0) & (true_int == 1)).sum().item()

    counts['tp'] = counts.get('tp', 0) + int(tp)
    counts['fp'] = counts.get('fp', 0) + int(fp)
    counts['fn'] = counts.get('fn', 0) + int(fn)
    return counts

# 计算precision, recall, f1
def compute_binary_f1(counts: dict):
    tp = counts.get('tp', 0)
    fp = counts.get('fp', 0)
    fn = counts.get('fn', 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1