import torch


def update_binary_counts(predictions: torch.Tensor, labels: torch.Tensor, counts: dict) -> dict:
    """
    累加二分类的 TP/FP/FN 计数。

    Args:
        predictions: 预测张量，取值为 {0,1}，形状 [N, 1] 或 [N]
        labels: 真实标签张量，取值为 {0,1}，形状 [N, 1] 或 [N]
        counts: 计数字典，包含键 'tp', 'fp', 'fn'

    Returns:
        更新后的 counts 字典
    """
    pred_int = predictions.long().view(-1)
    true_int = labels.long().view(-1)

    tp = ((pred_int == 1) & (true_int == 1)).sum().item()
    fp = ((pred_int == 1) & (true_int == 0)).sum().item()
    fn = ((pred_int == 0) & (true_int == 1)).sum().item()

    counts['tp'] = counts.get('tp', 0) + int(tp)
    counts['fp'] = counts.get('fp', 0) + int(fp)
    counts['fn'] = counts.get('fn', 0) + int(fn)
    return counts


def compute_binary_f1(counts: dict):
    """
    基于 TP/FP/FN 计数计算二分类的 Precision/Recall/F1。

    Args:
        counts: 包含 'tp', 'fp', 'fn' 的字典

    Returns:
        (precision, recall, f1)
    """
    tp = counts.get('tp', 0)
    fp = counts.get('fp', 0)
    fn = counts.get('fn', 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1