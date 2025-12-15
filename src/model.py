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


# ==================== Description Expert (单 MLP 版本) ====================

class DesExpertMoE(nn.Module):
    """
    Description Expert - 单 MLP 版本
    使用预训练 DeBERTa-v3-base（冻结参数）提取特征 + 单个 MLP
    """
    def __init__(self,
                 model_name='microsoft/deberta-v3-base',
                 hidden_dim=768,  # DeBERTa-v3-base 的输出维度是 768
                 expert_dim=64,
                 dropout=0.2,
                 device='cuda',
                 **kwargs):  # 忽略其他参数（兼容旧配置）

        super(DesExpertMoE, self).__init__()

        # 确定实际使用的设备
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim

        # DeBERTa-v3 模型和 Tokenizer (不微调，只用于特征提取)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.backbone_model = AutoModel.from_pretrained(model_name)
        self.backbone_model.eval()  # 设置为评估模式
        self.backbone_model = self.backbone_model.to(self.device)

        # 冻结 DeBERTa 参数，不进行微调
        for param in self.backbone_model.parameters():
            param.requires_grad = False

        # ========== 单个 MLP ==========
        # 768 → 512 → 256 → 128 → 64
        self.mlp = nn.Sequential(
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
        ).to(self.device)

        # ========== 分类头 ==========
        # 64 → 32 → 1
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
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

    def forward(self, description_text_list):
        """
        Args:
            description_text_list: List[str] or str
                - 每个元素是一个用户的简介文本

        Returns:
            expert_repr: [batch_size, expert_dim] - 专家表示
            bot_prob: [batch_size, 1] - bot 概率
        """
        # 确保输入是列表
        if isinstance(description_text_list, str):
            description_text_list = [description_text_list]

        # Step 1: 提取 DeBERTa 特征
        sentence_vectors = self._extract_deberta_features(description_text_list)  # [batch_size, 768]

        # Step 2: MLP 提取特征
        expert_repr = self.mlp(sentence_vectors)  # [batch_size, expert_dim]

        # Step 3: Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]

        return expert_repr, bot_prob

    def get_expert_repr(self, description_text_list):
        """只获取专家表示"""
        expert_repr, _ = self.forward(description_text_list)
        return expert_repr

    def print_expert_usage_stats(self):
        """兼容接口 - 单 MLP 无专家统计"""
        print(f"\n[Des Expert] 单 MLP 模型，无专家使用统计\n")


# ==================== Post Expert (单 MLP 版本) ====================

class PostExpert(nn.Module):
    """
    Post Expert - 单 MLP 版本
    使用预训练 DeBERTa-v3-base（冻结参数）提取特征 + 单个 MLP

    嵌入方式:
        对于用户 i 的 n 条推文:
        1. 每条推文被 DeBERTa 嵌入成 token-level 表示
        2. 平均每条推文的 tokens 得到该推文的向量
        3. 平均所有 n 条推文的向量，得到用户级表示 (768维)
        4. 单个 MLP 处理用户级表示
        5. 输出 64 维专家表示 + bot 概率
    """
    def __init__(self,
                 model_name='microsoft/deberta-v3-base',
                 hidden_dim=768,  # DeBERTa-v3-base 的输出维度是 768
                 expert_dim=64,
                 dropout=0.2,
                 device='cuda',
                 **kwargs):  # 忽略其他参数（兼容旧配置）

        super(PostExpert, self).__init__()

        # 确定实际使用的设备
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim

        # DeBERTa-v3 模型和 Tokenizer (不微调，只用于特征提取)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.backbone_model = AutoModel.from_pretrained(model_name)
        self.backbone_model.eval()  # 设置为评估模式
        self.backbone_model = self.backbone_model.to(self.device)

        # 冻结 DeBERTa 参数，不进行微调
        for param in self.backbone_model.parameters():
            param.requires_grad = False

        # ========== 单个 MLP ==========
        # 768 → 512 → 256 → 128 → 64
        self.mlp = nn.Sequential(
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
        ).to(self.device)

        # ========== 分类头 ==========
        # 64 → 32 → 1
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

    def _extract_user_post_embedding(self, user_posts):
        """
        提取单个用户的推文嵌入

        Args:
            user_posts: List[str] - 一个用户的推文列表

        Returns:
            user_embedding: [768] - 用户级推文嵌入
            is_valid: bool - 是否有有效推文
        """
        device = next(self.parameters()).device

        # 清理推文
        cleaned_posts = []
        for post in user_posts:
            post_str = str(post).strip()
            if post_str != '' and post_str.lower() != 'none':
                cleaned_posts.append(post_str)

        # 如果没有有效推文，返回零向量
        if len(cleaned_posts) == 0:
            return torch.zeros(self.hidden_dim, device=device), False

        # 使用 DeBERTa 提取特征
        with torch.no_grad():
            # Tokenize 所有推文
            encoded = self.tokenizer(
                cleaned_posts,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # 前向传播
            outputs = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [num_posts, seq_len, 768]

            # 对每条推文的所有 token 取平均，得到推文向量
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [num_posts, seq_len, 1]
            masked_hidden = hidden_states * attention_mask_expanded  # [num_posts, seq_len, 768]
            sum_hidden = masked_hidden.sum(dim=1)  # [num_posts, 768]
            sum_mask = attention_mask_expanded.sum(dim=1)  # [num_posts, 1]
            post_vectors = sum_hidden / sum_mask.clamp(min=1)  # [num_posts, 768]

            # 平均所有推文向量，得到用户级表示
            user_embedding = post_vectors.mean(dim=0)  # [768]

        return user_embedding, True

    def forward(self, posts_text_list):
        """
        Args:
            posts_text_list: List[List[str]]
                - 每个元素是一个用户的推文列表

        Returns:
            expert_repr: [batch_size, expert_dim] - 专家表示
            bot_prob: [batch_size, 1] - bot 概率
        """
        # Step 1: 提取每个用户的推文嵌入
        user_embeddings = []

        for user_posts in posts_text_list:
            user_emb, _ = self._extract_user_post_embedding(user_posts)
            user_embeddings.append(user_emb)

        # Stack: [batch_size, 768]
        user_embeddings = torch.stack(user_embeddings, dim=0)

        # Step 2: MLP 提取特征
        expert_repr = self.mlp(user_embeddings)  # [batch_size, expert_dim]

        # Step 3: Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]

        return expert_repr, bot_prob

    def get_expert_repr(self, posts_text_list):
        """只获取专家表示"""
        expert_repr, _ = self.forward(posts_text_list)
        return expert_repr

    def print_expert_usage_stats(self):
        """兼容接口 - 单 MLP 无专家统计"""
        print(f"\n[Post Expert] 单 MLP 模型，无专家使用统计\n")

# ==================== Cat Expert (类别属性专家 - 单 MLP 版本) ====================

class CatExpert(nn.Module):
    """
    Category Property Expert - 单 MLP 版本
    处理用户的类别型元数据（如 protected, verified, default_profile 等布尔属性）

    嵌入方式:
        1. 输入: 类别属性的 one-hot 向量 [batch_size, num_cat_features]
        2. MLP 学习表示: input_dim → 128 → 64 → expert_dim
        3. 分类头预测 bot 概率
    """
    def __init__(self,
                 input_dim=11,    # 类别属性数量（11个布尔属性）
                 expert_dim=64,   # 专家输出维度
                 dropout=0.2,
                 device='cuda'):

        super(CatExpert, self).__init__()

        # 确定实际使用的设备
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.expert_dim = expert_dim

        # ========== 单个 MLP ==========
        # input_dim → 128 → 64 → expert_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(64, expert_dim)
        ).to(self.device)

        # ========== 分类头 ==========
        # expert_dim → 32 → 1
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, cat_features):
        """
        Args:
            cat_features: [batch_size, input_dim] - 类别属性向量

        Returns:
            expert_repr: [batch_size, expert_dim] - 专家表示
            bot_prob: [batch_size, 1] - bot 概率
        """
        # MLP 提取特征
        expert_repr = self.mlp(cat_features)  # [batch_size, expert_dim]

        # Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]

        return expert_repr, bot_prob

    def get_expert_repr(self, cat_features):
        """只获取专家表示"""
        expert_repr, _ = self.forward(cat_features)
        return expert_repr

    def print_expert_usage_stats(self):
        """兼容接口 - 单 MLP 无专家统计"""
        print(f"\n[Cat Expert] 单 MLP 模型，无专家使用统计\n")


# ==================== Numerical Property Expert (单 MLP 版本) ====================

class NumExpert(nn.Module):
    """
    Numerical Property Expert - 单 MLP 版本
    处理用户的数值型元数据（如 followers_count, friends_count, statuses_count 等）

    嵌入方式:
        1. 输入: 数值属性向量 [batch_size, num_features]（已经过 z-score 标准化）
        2. MLP 学习表示: input_dim → 128 → 64 → expert_dim
        3. 分类头预测 bot 概率
    """
    def __init__(self,
                 input_dim=6,     # 数值属性数量（6个数值特征）
                 expert_dim=64,   # 专家输出维度
                 dropout=0.2,
                 device='cuda'):

        super(NumExpert, self).__init__()

        # 确定实际使用的设备
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.expert_dim = expert_dim

        # ========== 单个 MLP ==========
        # input_dim → 128 → 64 → expert_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(64, expert_dim)
        ).to(self.device)

        # ========== 分类头 ==========
        # expert_dim → 32 → 1
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, num_features):
        """
        Args:
            num_features: [batch_size, input_dim] - 数值属性向量（已标准化）

        Returns:
            expert_repr: [batch_size, expert_dim] - 专家表示
            bot_prob: [batch_size, 1] - bot 概率
        """
        # MLP 提取特征
        expert_repr = self.mlp(num_features)  # [batch_size, expert_dim]

        # Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]

        return expert_repr, bot_prob

    def get_expert_repr(self, num_features):
        """只获取专家表示"""
        expert_repr, _ = self.forward(num_features)
        return expert_repr

    def print_expert_usage_stats(self):
        """兼容接口 - 单 MLP 无专家统计"""
        print(f"\n[Num Expert] 单 MLP 模型，无专家使用统计\n")


class DesExpert (nn.Module):
    def __init__(self):
        super(DesExpert, self).__init__()
        self.model = DesExpertMoE()


# tweets专家模型 (保留原版本，兼容旧代码)
class  TweetsExpert(nn.Module):
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


# 图专家模型 (单 MLP 版本)
class GraphExpert(nn.Module):
    """
    Graph Expert Model - 单 MLP 版本
    使用 RGCN 处理异构图结构 + 单个 MLP
    节点特征从 node_embedding.py 生成的128维特征加载
    """
    def __init__(self,
                 num_nodes,
                 initial_node_features,  # 从 node_embedding.py 加载的初始节点特征 [num_nodes, 128]
                 num_relations=2,  # following (0) 和 follower (1) 两种关系
                 embedding_dim=128,  # RGCN 的隐藏层维度
                 expert_dim=64,
                 dropout=0.3,
                 device='cuda'):
        super(GraphExpert, self).__init__()

        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("需要安装 torch_geometric 才能使用 GraphExpert。\n"
                            "安装方法: pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html")

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.expert_dim = expert_dim
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'

        # 节点初始特征（从 node_embedding.py 加载），注册为 buffer
        initial_node_features = initial_node_features.to(self.device)
        self.register_buffer('initial_node_features', initial_node_features)

        input_dim = initial_node_features.shape[1]  # 应该是 128

        # ========== RGCN 层 ==========
        # 单层 RGCN: input_dim (128) -> embedding_dim (128)
        self.rgcn = RGCNConv(input_dim, embedding_dim, num_relations=num_relations, num_bases=num_relations)

        # ========== 后处理层 ==========
        # Linear + LeakyReLU
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)
        ).to(self.device)

        # ========== 单个 MLP ==========
        # embedding_dim → 128 → 64 → expert_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(64, expert_dim)
        ).to(self.device)

        # ========== 分类头 ==========
        # expert_dim → 32 → 1
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

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
        # Step 1: RGCN 图卷积（全图传播）
        x = self.initial_node_features
        x = self.rgcn(x, edge_index, edge_type)  # [num_nodes, embedding_dim]
        x = self.linear_relu_output1(x)  # [num_nodes, embedding_dim]

        # Step 2: 提取当前 batch 中节点的表示
        batch_node_features = x[node_indices]  # [batch_size, embedding_dim]

        # Step 3: MLP 提取特征
        expert_repr = self.mlp(batch_node_features)  # [batch_size, expert_dim]

        # Step 4: Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]

        return expert_repr, bot_prob

    def get_expert_repr(self, node_indices, edge_index, edge_type):
        """只获取专家表示，不计算 bot 概率"""
        expert_repr, _ = self.forward(node_indices, edge_index, edge_type)
        return expert_repr

    def print_expert_usage_stats(self):
        """兼容接口 - 单 MLP 无专家统计"""
        print(f"\n[Graph Expert] 单 MLP 模型，无专家使用统计\n")
