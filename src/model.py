import torch
from sentence_transformers import SentenceTransformer
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, DebertaV2Tokenizer
try:
    from torch_geometric.nn import RGCNConv
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError as _tg_err:
    RGCNConv = None
    _TORCH_GEOMETRIC_AVAILABLE = False


# ==================== MoE 版本的 Description Expert ====================

class DesExpertMoE(nn.Module):
    """
    Description Expert with Mixture of Experts (MoE) - Top-K 版本
    使用预训练 DeBERTa-v3-base（冻结参数）提取特征 + MoE(Gating + 多个MLP专家) 处理用户简介信息

    结构：
        DeBERTa (frozen) -> 768维特征
            ├── Gating Network -> 专家权重 [num_experts]
            ├── Expert 1 (MLP) -> 64维
            ├── Expert 2 (MLP) -> 64维
            ├── Expert 3 (MLP) -> 64维
            ├── Expert 4 (MLP) -> 64维
            └── Top-K 选择 (只选权重最大的 K 个专家)
        加权聚合 -> 64维专家表示 -> 分类头 -> bot概率
    """
    def __init__(self,
                 model_name='microsoft/deberta-v3-base',
                 hidden_dim=768,  # DeBERTa-v3-base 的输出维度是 768
                 expert_dim=64,
                 num_experts=4,   # MoE 中的专家数量（默认4个）
                 top_k=2,         # 每次选择的专家数量（默认Top-2）
                 dropout=0.2,
                 device='cuda'):

        super(DesExpertMoE, self).__init__()

        # 确定实际使用的设备
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # 确保 top_k 不超过专家数量
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim

        # 专家使用统计
        self.register_buffer('expert_usage_count', torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer('total_samples', torch.zeros(1, dtype=torch.long))

        # DeBERTa-v3 模型和 Tokenizer (不微调，只用于特征提取)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.backbone_model = AutoModel.from_pretrained(model_name)
        self.backbone_model.eval()  # 设置为评估模式
        self.backbone_model = self.backbone_model.to(self.device)

        # 冻结 DeBERTa 参数，不进行微调
        for param in self.backbone_model.parameters():
            param.requires_grad = False

        # ========== Gating Network ==========
        # 输入: 768维 DeBERTa 特征，输出: num_experts 个专家的权重
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_experts)  # 输出每个专家的 logit
        ).to(self.device)

        # ========== Expert Networks (MLP 专家) ==========
        # 每个专家都是一个 4 层 MLP: 768 → 512 → 256 → 128 → 64
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, expert_dim)
            )
            self.experts.append(expert_mlp.to(self.device))

        # ========== 分类头 ==========
        # 增强的分类头: 64 → 64 → 32 → 1（带 LayerNorm）
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 bot 概率
        ).to(self.device)

    def _extract_deberta_features(self, description_text_list):
        """
        使用 DeBERTa 提取文本特征

        Args:
            description_text_list: List[str] - 文本列表

        Returns:
            sentence_vectors: [batch_size, 768] - 句向量
        """
        device = next(self.parameters()).device

        # 清理简介文本
        cleaned_descriptions = []
        for desc in description_text_list:
            desc_str = str(desc).strip()
            if desc_str == '' or desc_str.lower() == 'none':
                desc_str = ''  # 空简介
            cleaned_descriptions.append(desc_str)

        # 使用 DeBERTa 提取特征（批量处理）
        with torch.no_grad():  # 不计算梯度，因为不微调 DeBERTa
            # Tokenize
            encoded = self.tokenizer(
                cleaned_descriptions,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # 前向传播
            outputs = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, 768]

            # 对每个样本的所有词向量取平均，得到句向量
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            masked_hidden = hidden_states * attention_mask_expanded  # [batch_size, seq_len, 768]
            sum_hidden = masked_hidden.sum(dim=1)  # [batch_size, 768]
            sum_mask = attention_mask_expanded.sum(dim=1)  # [batch_size, 1]
            sentence_vectors = sum_hidden / sum_mask.clamp(min=1)  # [batch_size, 768]

        return sentence_vectors

    def forward(self, description_text_list, return_gating_weights=False, return_expert_indices=False):
        """
        Args:
            description_text_list: List[str] or str
                - 每个元素是一个用户的简介文本
                - 例如: ["user1 description", "user2 description", ...]
            return_gating_weights: bool
                - 是否返回 gating 权重（用于分析）
            return_expert_indices: bool
                - 是否返回选中的专家索引

        Returns:
            expert_repr: [batch_size, expert_dim] - 聚合后的专家表示
            bot_prob: [batch_size, 1] - bot 概率
            gating_weights: [batch_size, num_experts] - (可选) 各专家权重
            expert_indices: [batch_size, top_k] - (可选) 选中的专家索引
        """
        # 确保输入是列表
        if isinstance(description_text_list, str):
            description_text_list = [description_text_list]

        batch_size = len(description_text_list)

        # Step 1: 提取 DeBERTa 特征
        sentence_vectors = self._extract_deberta_features(description_text_list)  # [batch_size, 768]

        # Step 2: Gating Network 计算专家权重
        gating_logits = self.gating_network(sentence_vectors)  # [batch_size, num_experts]
        gating_weights = F.softmax(gating_logits, dim=-1)  # [batch_size, num_experts]

        # Step 3: Top-K 选择专家
        # 获取权重最大的 top_k 个专家的索引和权重
        topk_weights, topk_indices = torch.topk(gating_weights, self.top_k, dim=-1)
        # topk_weights: [batch_size, top_k]
        # topk_indices: [batch_size, top_k]

        # 重新归一化 top-k 权重
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # [batch_size, top_k]

        # 统计专家使用次数（仅在训练/评估时）
        if self.training or not return_gating_weights:
            with torch.no_grad():
                for i in range(batch_size):
                    for expert_idx in topk_indices[i]:
                        self.expert_usage_count[expert_idx] += 1
                self.total_samples += batch_size

        # Step 4: 只计算被选中的专家的输出
        # 为了效率，我们还是计算所有专家，但只聚合 top-k
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(sentence_vectors)  # [batch_size, expert_dim]
            expert_outputs.append(expert_out)

        # Stack: [batch_size, num_experts, expert_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Step 5: Top-K 加权聚合
        # 根据 topk_indices 选择对应的专家输出
        batch_indices = torch.arange(batch_size, device=sentence_vectors.device).unsqueeze(1).expand(-1, self.top_k)
        selected_expert_outputs = expert_outputs[batch_indices, topk_indices]  # [batch_size, top_k, expert_dim]

        # 加权求和
        topk_weights_expanded = topk_weights.unsqueeze(-1)  # [batch_size, top_k, 1]
        expert_repr = (selected_expert_outputs * topk_weights_expanded).sum(dim=1)  # [batch_size, expert_dim]

        # Step 6: Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]

        # 返回结果
        results = [expert_repr, bot_prob]
        if return_gating_weights:
            results.append(gating_weights)
        if return_expert_indices:
            results.append(topk_indices)

        if len(results) == 2:
            return expert_repr, bot_prob
        elif len(results) == 3:
            return tuple(results)
        else:
            return tuple(results)

    def get_expert_repr(self, description_text_list):
        """只获取专家表示，用于 gating network 使用"""
        expert_repr, _ = self.forward(description_text_list)
        return expert_repr

    def get_gating_weights(self, description_text_list):
        """获取 gating 权重，用于分析专家使用情况"""
        return self.forward(description_text_list, return_gating_weights=True)[2]

    def get_expert_usage_stats(self):
        """
        获取专家使用统计信息

        Returns:
            dict: 包含每个专家的使用次数和使用率
        """
        total = self.total_samples.item()
        if total == 0:
            return {
                'total_samples': 0,
                'expert_counts': [0] * self.num_experts,
                'expert_rates': [0.0] * self.num_experts
            }

        counts = self.expert_usage_count.cpu().tolist()
        rates = [count / total * 100 for count in counts]  # 百分比

        return {
            'total_samples': total,
            'expert_counts': counts,
            'expert_rates': rates
        }

    def reset_expert_usage_stats(self):
        """重置专家使用统计"""
        self.expert_usage_count.zero_()
        self.total_samples.zero_()

    def print_expert_usage_stats(self):
        """打印专家使用统计信息"""
        stats = self.get_expert_usage_stats()
        print(f"\n{'='*60}")
        print(f"专家使用统计 (Top-{self.top_k} 选择)")
        print(f"{'='*60}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"{'专家':<10} {'使用次数':<15} {'使用率':<15}")
        print(f"{'-'*60}")
        for i in range(self.num_experts):
            print(f"Expert {i+1:<3} {stats['expert_counts'][i]:<15} {stats['expert_rates'][i]:<14.2f}%")
        print(f"{'='*60}\n")

# tweets专家模型
class TweetsExpert(nn.Module):
    """
    Tweets Expert Model
    使用预训练 RoBERTa（不微调）提取特征 + MLP 处理推文信息，对用户的多条推文做平均聚合
    """
    def __init__(self, 
                 roberta_model_name='distilroberta-base',
                 hidden_dim=768,
                 expert_dim=64,
                 dropout=0.1,
                 device='cuda'):
        super(TweetsExpert, self).__init__()
        
        # RoBERTa 模型和 Tokenizer (不微调，只用于特征提取)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        self.roberta_model = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_model.eval()  # 设置为评估模式，不计算梯度
        # 冻结 RoBERTa 参数，不进行微调
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        # 移动到指定设备
        self.roberta_model = self.roberta_model.to(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')

        self.hidden_dim = hidden_dim
        
        # MLP 网络
        # 从 RoBERTa 的 768维句向量到 64维 Expert Representation(768->256->128->64)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, expert_dim)
        )
        
        # Bot Probability 预测头
        # 从 64维 Expert Representation 到 1维 bot 概率（64->32->1)
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 bot 概率
        )
    
    def forward(self, tweets_text_list):
        """
        Args:
            tweets_text_list: List[List[str]]
                - 每个元素是一个用户的推文列表（字符串列表）
                - 例如: [["tweet1 text", "tweet2 text", ...], ["tweet1 text", ...], ...]
        """
        batch_size = len(tweets_text_list) # 前向传播时当前batch的用户数量
        batch_expert_reprs = [] # 每个用户的tweets专家表示
        device = next(self.parameters()).device
        
        # 对每个用户的多条推文进行处理
        for user_idx in range(batch_size):
            user_tweets = tweets_text_list[user_idx]  # 该用户的所有推文（字符串列表）
            
            # 如果用户没有推文（空列表），使用零向量
            if len(user_tweets) == 0 or (len(user_tweets) == 1 and user_tweets[0] == ''):
                expert_repr = torch.zeros(self.mlp[-1].out_features, device=device)
                batch_expert_reprs.append(expert_repr)
                continue
            
            # 清理推文文本
            cleaned_tweets = []
            for tweet_text in user_tweets:
                tweet_text = str(tweet_text).strip()
                if tweet_text != '' and tweet_text != 'None':
                    cleaned_tweets.append(tweet_text)
            
            if len(cleaned_tweets) == 0:
                user_expert_repr = torch.zeros(self.mlp[-1].out_features, device=device)
                batch_expert_reprs.append(user_expert_repr)
                continue
            
            # 批量处理该用户的所有推文（提高效率）
            with torch.no_grad():  # 不计算梯度，因为不微调 RoBERTa
                # Tokenize 所有推文
                encoded = self.roberta_tokenizer(
                    cleaned_tweets,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # 移动到正确设备
                input_ids = encoded['input_ids'].to(self.roberta_model.device)
                attention_mask = encoded['attention_mask'].to(self.roberta_model.device)

                # 使用 RoBERTa 提取特征
                outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
                # outputs.last_hidden_state: [num_tweets, seq_len, 768]
                hidden_states = outputs.last_hidden_state  # [num_tweets, seq_len, 768]
                
                # 对每条推文的所有词向量取平均，得到句向量
                # 使用 attention_mask 来正确计算平均值（忽略 padding）
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [num_tweets, seq_len, 1]
                masked_hidden = hidden_states * attention_mask_expanded  # [num_tweets, seq_len, 768]
                sum_hidden = masked_hidden.sum(dim=1)  # [num_tweets, 768] - 每条推文的词向量和
                sum_mask = attention_mask_expanded.sum(dim=1)  # [num_tweets, 1] - 每条推文的有效词数
                sentence_vectors = sum_hidden / sum_mask.clamp(min=1)  # [num_tweets, 768] - 平均得到句向量
            
            # 将句向量移动到模型设备
            sentence_vectors = sentence_vectors.to(device)

            # MLP: 768维 → 64维 Expert Representation（批量处理）
            tweet_expert_reprs = self.mlp(sentence_vectors)  # [num_tweets, 64]
            
            # 对所有推文的专家表示做平均聚合
            user_expert_repr = torch.mean(tweet_expert_reprs, dim=0)  # [64] - 平均聚合
            
            batch_expert_reprs.append(user_expert_repr)
        
        # 将该batch中每个用户的表示拼成张量（batch_size*64的矩阵）
        expert_repr = torch.stack(batch_expert_reprs, dim=0)  # [batch_size, 64]
        
        # Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]
        
        return expert_repr, bot_prob

    # 只获取专家表示，不计算 bot 概率
    def get_expert_repr(self, tweets_text_list):
        expert_repr, _ = self.forward(tweets_text_list)
        return expert_repr


# 图专家模型
class GraphExpert(nn.Module):
    """
    Graph Expert Model
    使用 RGCN 处理异构图结构，节点特征由其他专家的表示聚合而来
    """
    def __init__(self,
                 num_nodes,
                 initial_node_features,  # 从其他专家聚合得到的初始节点特征 [num_nodes, feature_dim]
                 num_relations=2,  # following (0) 和 follower (1) 两种关系
                 hidden_dim=128,
                 expert_dim=64,
                 num_layers=2,
                 dropout=0.3,
                 device='cuda'):
        super(GraphExpert, self).__init__()

        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("需要安装 torch_geometric 才能使用 GraphExpert。\n"
                            "安装方法: pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html")

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_layers = num_layers
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'

        # 节点初始特征（从其他专家聚合得到），注册为 buffer，随模型保存/加载
        initial_node_features = initial_node_features.to(self.device)
        self.register_buffer('initial_node_features', initial_node_features)

        input_dim = initial_node_features.shape[1]

        # RGCN 层
        self.rgcn_layers = nn.ModuleList()

        if num_layers == 1:
            # 如果只有一层，直接从输入维度 -> expert_dim
            self.rgcn_layers.append(
                RGCNConv(input_dim, expert_dim, num_relations=num_relations, num_bases=num_relations)
            )
        else:
            # 第一层: input_dim -> hidden_dim
            self.rgcn_layers.append(
                RGCNConv(input_dim, hidden_dim, num_relations=num_relations, num_bases=num_relations)
            )
            # 中间层: hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                self.rgcn_layers.append(
                    RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_relations)
                )
            # 最后一层: hidden_dim -> expert_dim
            self.rgcn_layers.append(
                RGCNConv(hidden_dim, expert_dim, num_relations=num_relations, num_bases=num_relations)
            )

        self.dropout = nn.Dropout(dropout)

        # Bot Probability 预测头（与其他专家保持一致）
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, node_indices, edge_index, edge_type):
        """
        Args:
            node_indices: [batch_size] - 当前 batch 中节点的索引（在完整图中的索引）
            edge_index: [2, num_edges] - 图的边索引
            edge_type: [num_edges] - 每条边的类型（0: following, 1: follower）

        Returns:
            expert_repr: [batch_size, expert_dim] - 专家表示（64维）
            bot_prob: [batch_size, 1] - bot 概率
        """
        # 通过 RGCN 层进行图卷积（全图传播，提取所有节点的表示）
        x = self.initial_node_features

        for i, rgcn_layer in enumerate(self.rgcn_layers):
            x = rgcn_layer(x, edge_index, edge_type)
            if i < len(self.rgcn_layers) - 1:  # 最后一层不加激活和dropout
                x = F.relu(x)
                x = self.dropout(x)

        # 提取当前 batch 中节点的表示 [batch_size, expert_dim]
        batch_expert_repr = x[node_indices]

        # Bot Probability 预测
        bot_prob = self.bot_classifier(batch_expert_repr)  # [batch_size, 1]

        return batch_expert_repr, bot_prob

    def get_expert_repr(self, node_indices, edge_index, edge_type):
        """只获取专家表示，不计算 bot 概率（用于 gating network）"""
        expert_repr, _ = self.forward(node_indices, edge_index, edge_type)
        return expert_repr
