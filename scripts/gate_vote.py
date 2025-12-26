"""
Voting Gate - 投票机制门控
使用少数服从多数的投票机制聚合多个专家模型的预测结果

投票规则:
1. 每个专家对样本进行二分类预测（p > 0.5 为 bot，否则为 human）
2. 统计预测为 bot 的专家数量和预测为 human 的专家数量
3. 多数票决定最终预测结果
4. 支持 mask 机制，缺失数据的专家不参与投票

例如: 对于样本 a，3 个专家预测为 human，2 个专家预测为 bot
      则 3 > 2，最终结果为 human
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import update_binary_counts, compute_binary_f1
from scripts.gate import ExpertEmbeddingDataset


class VotingGate(nn.Module):
    """
    投票门控机制（无需训练）
    
    使用少数服从多数的投票规则:
    - 每个专家的预测概率 > 0.5 视为预测 bot (1)
    - 每个专家的预测概率 <= 0.5 视为预测 human (0)
    - 统计投票数，多数票决定最终结果
    - 平票时（如 2:2），默认预测为 bot（保守策略）
    """
    
    def __init__(self, num_experts=5, threshold=0.5):
        """
        Args:
            num_experts: 专家数量（默认 5: cat, num, graph, des, post）
            threshold: 预测阈值（默认 0.5）
        """
        super(VotingGate, self).__init__()
        self.num_experts = num_experts
        self.threshold = threshold
        self.expert_names = ['cat', 'num', 'graph', 'des', 'post']
    
    def forward(self, p_cat, p_num, p_graph, p_des, p_post, expert_mask=None):
        """
        使用投票机制聚合各专家的预测
        
        Args:
            p_cat, p_num, p_graph, p_des, p_post: [batch_size, 1] - 各专家的预测概率
            expert_mask: [batch_size, 5] - 专家有效性掩码
                         顺序: [cat, num, graph, des, post]
                         1 = 有效，0 = 缺失（该专家不参与投票）
                         如果为 None，则假设所有专家都有效
        
        Returns:
            predictions: [batch_size, 1] - 最终预测结果（0 或 1）
            vote_counts: dict - 投票统计信息
        """
        batch_size = p_cat.size(0)
        device = p_cat.device
        
        # 将各专家的预测概率堆叠 [batch_size, 5]
        probs = torch.cat([p_cat, p_num, p_graph, p_des, p_post], dim=-1)  # [batch_size, 5]
        
        # 将概率转换为二分类预测 (> threshold 为 bot=1)
        expert_preds = (probs > self.threshold).float()  # [batch_size, 5]
        
        # 应用 mask（缺失的专家不参与投票）
        if expert_mask is not None:
            expert_mask = expert_mask.float()
            # 有效专家数量
            valid_counts = expert_mask.sum(dim=-1, keepdim=True)  # [batch_size, 1]
            # 预测为 bot 的有效专家数量
            bot_votes = (expert_preds * expert_mask).sum(dim=-1, keepdim=True)  # [batch_size, 1]
        else:
            valid_counts = torch.full((batch_size, 1), self.num_experts, device=device)
            bot_votes = expert_preds.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        
        # 预测为 human 的专家数量
        human_votes = valid_counts - bot_votes  # [batch_size, 1]
        
        # 投票决策: bot_votes > human_votes 则预测为 bot
        # 平票时（bot_votes == human_votes），默认预测为 bot（保守策略）
        predictions = (bot_votes >= human_votes).float()  # [batch_size, 1]
        
        # 统计信息
        vote_counts = {
            'bot_votes': bot_votes,
            'human_votes': human_votes,
            'valid_experts': valid_counts
        }
        
        return predictions, vote_counts
    
    def get_vote_details(self, p_cat, p_num, p_graph, p_des, p_post, expert_mask=None):
        """
        获取详细的投票信息
        
        Returns:
            dict: 包含每个专家的预测和投票结果
        """
        probs = torch.cat([p_cat, p_num, p_graph, p_des, p_post], dim=-1)
        expert_preds = (probs > self.threshold).float()
        
        predictions, vote_counts = self.forward(p_cat, p_num, p_graph, p_des, p_post, expert_mask)
        
        return {
            'expert_probs': {
                'cat': p_cat,
                'num': p_num,
                'graph': p_graph,
                'des': p_des,
                'post': p_post
            },
            'expert_preds': {
                'cat': expert_preds[:, 0:1],
                'num': expert_preds[:, 1:2],
                'graph': expert_preds[:, 2:3],
                'des': expert_preds[:, 3:4],
                'post': expert_preds[:, 4:5]
            },
            'final_prediction': predictions,
            'vote_counts': vote_counts
        }


class VotingEvaluator:
    """
    投票门控评估器
    
    由于投票机制不需要训练，只需要评估
    """
    
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
    
    def evaluate(self):
        """评估投票门控的性能"""
        self.model.eval()
        
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        
        # 统计各专家的投票情况
        total_bot_votes = 0
        total_human_votes = 0
        total_valid_experts = 0
        
        # 统计各专家的准确率
        expert_correct = {'cat': 0, 'num': 0, 'graph': 0, 'des': 0, 'post': 0}
        expert_total = {'cat': 0, 'num': 0, 'graph': 0, 'des': 0, 'post': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                p_cat = batch['p_cat'].to(self.device)
                p_num = batch['p_num'].to(self.device)
                p_graph = batch['p_graph'].to(self.device)
                p_des = batch['p_des'].to(self.device)
                p_post = batch['p_post'].to(self.device)
                expert_mask = batch['expert_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 投票预测
                predictions, vote_counts = self.model(
                    p_cat, p_num, p_graph, p_des, p_post, expert_mask
                )
                
                # 统计
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                update_binary_counts(predictions, labels, counts)
                
                # 投票统计
                total_bot_votes += vote_counts['bot_votes'].sum().item()
                total_human_votes += vote_counts['human_votes'].sum().item()
                total_valid_experts += vote_counts['valid_experts'].sum().item()
                
                # 各专家准确率统计
                probs_list = [p_cat, p_num, p_graph, p_des, p_post]
                names = ['cat', 'num', 'graph', 'des', 'post']
                for i, (prob, name) in enumerate(zip(probs_list, names)):
                    pred = (prob > 0.5).float()
                    mask = expert_mask[:, i:i+1]
                    expert_correct[name] += ((pred == labels) * mask).sum().item()
                    expert_total[name] += mask.sum().item()
        
        # 计算指标
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)
        
        # 各专家准确率
        expert_acc = {}
        for name in names:
            if expert_total[name] > 0:
                expert_acc[name] = expert_correct[name] / expert_total[name]
            else:
                expert_acc[name] = 0.0
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_bot_votes': total_bot_votes / total,
            'avg_human_votes': total_human_votes / total,
            'avg_valid_experts': total_valid_experts / total,
            'expert_accuracy': expert_acc
        }


def main():
    """
    评估投票门控的主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Voting Gate')
    parser.add_argument('--embedding_dir', type=str, default='../../autodl-fs/labeled_embedding',
                        help='专家 embedding 文件目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--threshold', type=float, default=0.5, help='预测阈值')
    
    args = parser.parse_args()
    
    # 设备
    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载专家 embedding 数据...")
    train_dataset = ExpertEmbeddingDataset('train', args.embedding_dir)
    val_dataset = ExpertEmbeddingDataset('val', args.embedding_dir)
    test_dataset = ExpertEmbeddingDataset('test', args.embedding_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建投票门控模型（无需训练）
    model = VotingGate(num_experts=5, threshold=args.threshold)
    
    # 评估器
    evaluator = VotingEvaluator(model, test_loader, device)
    
    # 评估各数据集
    print(f"\n{'=' * 60}")
    print("投票门控 (Voting Gate) 评估结果")
    print(f"{'=' * 60}")
    print(f"投票规则: 少数服从多数，平票时预测为 bot")
    print(f"预测阈值: {args.threshold}")
    
    for split, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        evaluator.test_loader = loader
        results = evaluator.evaluate()
        
        print(f"\n[{split}]")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1']:.4f}")
        print(f"  平均 Bot 票数:   {results['avg_bot_votes']:.2f}")
        print(f"  平均 Human 票数: {results['avg_human_votes']:.2f}")
        print(f"  平均有效专家数:  {results['avg_valid_experts']:.2f}")
        print(f"  各专家准确率:")
        for name, acc in results['expert_accuracy'].items():
            print(f"    {name}: {acc:.4f}")
    
    print(f"\n{'=' * 60}\n")
    
    return results


if __name__ == '__main__':
    main()
