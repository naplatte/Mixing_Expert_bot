import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, DebertaV2Tokenizer
try:
    from torch_geometric.nn import RGCNConv
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError as _tg_err:
    RGCNConv = None
    _TORCH_GEOMETRIC_AVAILABLE = False


class MLP(nn.Module):
    """
    简单的两层 MLP，用于特征降维
    input_size -> hidden_size -> output_size
    """
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# description专家模型
class DesExpert(nn.Module):
    """
    Description Expert Model
    使用预训练 DeBERTa-v3（冻结参数）提取特征 + MLP 处理用户简介信息
    """
    def __init__(self,
                 model_name='microsoft/deberta-v3-base',
                 hidden_dim=768,
                 expert_dim=64,
                 dropout=0.1,
                 device='cuda'):

        super(DesExpert, self).__init__()
        
        # 确定实际使用的设备
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'

        # DeBERTa-v3 模型和 Tokenizer (不微调，只用于特征提取)
        # 显式使用慢速 DebertaV2 tokenizer，避免 HuggingFace 自动尝试 fast tokenizer -> tiktoken 转换
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name, use_fast=False)
        self.backbone_model = AutoModel.from_pretrained(model_name)

        # 获取模型的实际 hidden size
        actual_hidden_size = self.backbone_model.config.hidden_size

        self.backbone_model.eval()  # 设置为评估模式
        # 冻结 DeBERTa-v3 参数，不进行微调
        for param in self.backbone_model.parameters():
            param.requires_grad = False
        # 移动到指定设备
        self.backbone_model = self.backbone_model.to(self.device)

        self.hidden_dim = actual_hidden_size

        # MLP 网络
        # 从 DeBERTa-v3 的句向量到 64维 Expert Representation
        self.mlp = nn.Sequential(
            nn.Linear(actual_hidden_size, 512),
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
        ).to(self.device)

        # Bot Probability 预测头 - 增强版本
        # 从 64维 Expert Representation 到 1维 bot 概率
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

    def forward(self, description_text_list):
        """
        Args:
            description_text_list: List[str] or str
                - 每个元素是一个用户的简介文本
                - 例如: ["user1 description", "user2 description", ...]
        """
        # 确保输入是列表
        if isinstance(description_text_list, str):
            description_text_list = [description_text_list]

        device = next(self.parameters()).device

        # 清理简介文本
        cleaned_descriptions = []
        for desc in description_text_list:
            desc_str = str(desc).strip()
            if desc_str == '' or desc_str.lower() == 'none':
                desc_str = ''  # 空简介
            cleaned_descriptions.append(desc_str)

        # 使用 DeBERTa-v3 提取特征（批量处理）
        with torch.no_grad():  # 不计算梯度，因为不微调 DeBERTa-v3
            # Tokenize 所有简介
            encoded = self.tokenizer(
                cleaned_descriptions,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # 移动到正确设备
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # 使用 DeBERTa-v3 提取特征
            outputs = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

            # 对每个简介的所有词向量取平均，得到句向量
            # 使用 attention_mask 来正确计算平均值（忽略 padding）
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            masked_hidden = hidden_states * attention_mask_expanded  # [batch_size, seq_len, hidden_dim]
            sum_hidden = masked_hidden.sum(dim=1)  # [batch_size, hidden_dim]
            sum_mask = attention_mask_expanded.sum(dim=1)  # [batch_size, 1]
            sentence_vectors = sum_hidden / sum_mask.clamp(min=1)  # [batch_size, hidden_dim]

        # 确保 sentence_vectors 在正确的设备上（退出 no_grad 后显式检查）
        sentence_vectors = sentence_vectors.to(device)

        # MLP: hidden_dim → 64维 Expert Representation
        expert_repr = self.mlp(sentence_vectors)  # 专家表示，shape:[batch_size, 64]

        # Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # shape:[batch_size, 1]
        
        return expert_repr, bot_prob

    # 只获取专家表示，用于 gating network 使用
    def get_expert_repr(self, description_text_list):
        expert_repr, _ = self.forward(description_text_list)
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
