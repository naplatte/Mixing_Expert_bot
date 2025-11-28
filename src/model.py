import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel, AutoTokenizer
from torch_geometric.nn import RGCNConv
_TORCH_GEOMETRIC_AVAILABLE = True

# description专家模型
class DesExpert(nn.Module):
    def __init__(self, 
                 bert_model_name='bert-base-uncased',
                 hidden_dim=768,
                 expert_dim=64,
                 dropout=0.1,
                 freeze_bert=True):

        super(DesExpert, self).__init__()
        
        # whether to freeze BERT parameters (use BERT as fixed feature extractor)
        self.freeze_bert = freeze_bert

        # BERT Encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        if self.freeze_bert:
            # 冻结bert参数，设置eval模式
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()

        self.hidden_dim = hidden_dim
        
        # MLP 网络
        # 从 BERT 的 768维 [CLS] token 到 64维 Expert Representation(768->256->128->64)
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

    # input_ids - 由tokenizer生成，形状[batch_size,seq_len]，即为该batch中每个句子的 token id 序列（每个句子的长度都是固定的 - 固定为seq_len）,实际为batch_size * seq_len形状的矩阵，即batch_size个句子的初始token
    # attention_mask:[batch_size,seq_len]，mask 掩码（1表示有效，0表示padding）（短句子会被填充至seq_len长度，填充部分的mask掩码为0，文本部分为1），告诉 BERT 哪些 token 是 padding（0 表示忽略）
    def forward(self, input_ids, attention_mask=None):
        # 以下两步本质就是从初始输入input_ids获取到最终句向量cls_embedding
        if self.freeze_bert:
            # use BERT as frozen feature extractor to save memory and computation
            with torch.no_grad():
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = bert_outputs.pooler_output  # [batch_size, 768]

        # MLP: 768维 → 64维 Expert Representation
        expert_repr = self.mlp(cls_embedding)  # 专家表示，shape:[batch_size, 64]
        
        # Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # shape:[batch_size, 1]
        
        return expert_repr, bot_prob

    # 只获取专家表示，用于 gating network 使用
    def get_expert_repr(self, input_ids, attention_mask=None):
        if self.freeze_bert:
            with torch.no_grad():
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_outputs.pooler_output
        expert_repr = self.mlp(cls_embedding)
        return expert_repr

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


# 图结构专家模型
class GraphExpert(nn.Module):
    # 输入: 图结构 (edge_index, edge_type) 和节点特征
    def __init__(self,
                 num_nodes,
                 node_features,
                 num_relations=2,  # following (0) 和 follower (1) 两种关系
                 hidden_dim=128,
                 expert_dim=64,
                 num_layers=2,
                 dropout=0.1,
                 device='cuda'):
        super(GraphExpert, self).__init__()
        
        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("需要安装 torch_geometric 才能使用 GraphExpert。安装方法: pip install torch-geometric")
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.device = device
        
        # 节点结构特征（预先计算），注册为 buffer，随模型保存/加载
        node_features = node_features.to(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.register_buffer('node_features', node_features)
        
        # RGCN 层
        self.rgcn_layers = nn.ModuleList()
        input_dim = node_features.shape[1]
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
        if num_layers > 1:
            self.rgcn_layers.append(
                RGCNConv(hidden_dim, expert_dim, num_relations=num_relations, num_bases=num_relations)
            )
        else:
            # 如果只有一层，直接从输入维度 -> expert_dim
            self.rgcn_layers[0] = RGCNConv(input_dim, expert_dim, num_relations=num_relations, num_bases=num_relations)
        
        self.dropout = nn.Dropout(dropout)
        
        # Bot Probability 预测头
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
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
            expert_repr: [batch_size, expert_dim] - 专家表示
            bot_prob: [batch_size, 1] - bot 概率
        """
        # 通过 RGCN 层进行图卷积
        x = self.node_features
        for i, rgcn_layer in enumerate(self.rgcn_layers):
            x = rgcn_layer(x, edge_index, edge_type)
            if i < len(self.rgcn_layers) - 1:  # 最后一层不加激活和dropout
                x = F.relu(x)
                x = self.dropout(x)
        
        # 提取当前 batch 中节点的表示 [batch_size, expert_dim]
        batch_node_features = x[node_indices]
        
        # Bot Probability 预测
        bot_prob = self.bot_classifier(batch_node_features)  # [batch_size, 1]
        
        return batch_node_features, bot_prob
    
    def get_expert_repr(self, node_indices, edge_index, edge_type):
        """只获取专家表示，不计算 bot 概率"""
        expert_repr, _ = self.forward(node_indices, edge_index, edge_type)
        return expert_repr
    
    def get_all_node_repr(self, edge_index, edge_type):
        """
        获取所有节点的专家表示（用于图嵌入）
        Returns:
            all_repr: [num_nodes, expert_dim]
        """
        all_node_features = self.node_embedding(torch.arange(self.num_nodes, device=self.device))
        x = all_node_features
        for i, rgcn_layer in enumerate(self.rgcn_layers):
            x = rgcn_layer(x, edge_index, edge_type)
            if i < len(self.rgcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


# 门控聚合器：聚合多个专家的输出
class ExpertGatedAggregator(nn.Module):
    """
    门控网络：根据各专家的表示和概率，动态计算权重并聚合
    """
    def __init__(self, num_experts=3, expert_dim=64, hidden_dim=128):
        super(ExpertGatedAggregator, self).__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        
        # 门控网络：根据专家表示和概率计算权重
        # 输入：专家表示拼接 + 专家概率拼接
        # 输出：每个专家的权重
        self.gate_network = nn.Sequential(
            nn.Linear(expert_dim * num_experts + num_experts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, expert_reprs, expert_probs, active_mask):
        """
        Args:
            expert_reprs: List of [batch_size, expert_dim] tensors
                - 每个专家的表示
            expert_probs: List of [batch_size, 1] tensors
                - 每个专家的概率预测
            active_mask: [batch_size, num_experts]
                - 1表示专家可用，0表示不可用（例如：没有description或tweets）
        
        Returns:
            final_prob: [batch_size, 1] - 最终聚合的概率
            gate_weights: [batch_size, num_experts] - 每个专家的权重
        """
        batch_size = expert_reprs[0].shape[0]
        
        # 拼接所有专家的表示 [batch_size, expert_dim * num_experts]
        expert_reprs_concat = torch.cat(expert_reprs, dim=1)
        
        # 拼接所有专家的概率 [batch_size, num_experts]
        expert_probs_concat = torch.cat(expert_probs, dim=1)
        
        # 门控网络输入：专家表示 + 专家概率
        gate_input = torch.cat([expert_reprs_concat, expert_probs_concat], dim=1)  # [batch_size, expert_dim * num_experts + num_experts]
        
        # 计算门控权重 [batch_size, num_experts]
        gate_weights = self.gate_network(gate_input)
        
        # 应用 active_mask：不可用的专家权重设为0
        gate_weights = gate_weights * active_mask
        
        # 重新归一化（避免除零）
        gate_weights_sum = gate_weights.sum(dim=1, keepdim=True)
        gate_weights = gate_weights / (gate_weights_sum + 1e-8)
        
        # 加权聚合专家概率
        weighted_prob = (gate_weights * expert_probs_concat).sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        return weighted_prob, gate_weights
