import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, DebertaV2Tokenizer

try:
    from torch_geometric.nn import RGCNConv
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    RGCNConv = None
    _TORCH_GEOMETRIC_AVAILABLE = False

# 专家模型基类，包含公共的MLP和分类头
class BaseExpert(nn.Module):
    def __init__(self, input_dim, expert_dim=64, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.expert_dim = expert_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, expert_dim)
        ).to(self.device)

        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

# 文本专家基类(Des/Post)， 使用DeBERTa提取特征
class TextExpertBase(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base', hidden_dim=768,
                 expert_dim=64, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim

        # DeBERTa (冻结参数)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        self.backbone = self.backbone.to(self.device)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # MLP: 768 -> 512 -> 256 -> 128 -> 64
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

        # 分类头
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

    def _encode_texts(self, texts):
        """编码文本列表，返回句向量"""
        device = next(self.parameters()).device
        cleaned = [str(t).strip() if str(t).strip().lower() != 'none' else '' for t in texts]

        with torch.no_grad():
            encoded = self.tokenizer(cleaned, max_length=128, padding=True,
                                     truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state

            # 平均池化
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1)
            return sum_hidden / sum_mask

    def get_expert_repr(self, *args, **kwargs):
        expert_repr, _ = self.forward(*args, **kwargs)
        return expert_repr


class DesExpertMoE(TextExpertBase):
    """描述专家 - 处理用户简介"""

    def forward(self, description_list):
        if isinstance(description_list, str):
            description_list = [description_list]
        vectors = self._encode_texts(description_list)
        expert_repr = self.mlp(vectors)
        bot_prob = self.bot_classifier(expert_repr)
        return expert_repr, bot_prob


class PostExpert(TextExpertBase):
    """推文专家 - 处理用户推文(注意力聚合)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 注意力网络
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        ).to(self.device)

    def _extract_user_embedding(self, user_posts):
        """提取单个用户的推文嵌入"""
        device = next(self.parameters()).device
        cleaned = [str(p).strip() for p in user_posts
                   if str(p).strip() and str(p).strip().lower() != 'none']

        if not cleaned:
            return torch.zeros(self.hidden_dim, device=device), False

        post_vectors = self._encode_texts(cleaned)

        # 注意力加权聚合
        attn_scores = self.attention(post_vectors).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(-1)
        return (attn_weights * post_vectors).sum(dim=0), True

    def forward(self, posts_list):
        embeddings = []
        for user_posts in posts_list:
            emb, _ = self._extract_user_embedding(user_posts)
            embeddings.append(emb)

        user_embs = torch.stack(embeddings, dim=0)
        expert_repr = self.mlp(user_embs)
        bot_prob = self.bot_classifier(expert_repr)
        return expert_repr, bot_prob


class CatExpert(BaseExpert):
    def __init__(self, input_dim=3, expert_dim=64, dropout=0.2, device='cuda'):
        super().__init__(input_dim, expert_dim, dropout, device)

    def forward(self, cat_features):
        expert_repr = self.mlp(cat_features)
        bot_prob = self.bot_classifier(expert_repr)
        return expert_repr, bot_prob


class NumExpert(BaseExpert):
    def __init__(self, input_dim=5, expert_dim=64, dropout=0.2, device='cuda'):
        super().__init__(input_dim, expert_dim, dropout, device)

    def forward(self, num_features):
        expert_repr = self.mlp(num_features)
        bot_prob = self.bot_classifier(expert_repr)
        return expert_repr, bot_prob


class GraphExpert(nn.Module):
    """图专家 - 使用RGCN处理图结构"""

    def __init__(self, num_nodes, embedding_dir='../../autodl-fs/node_embedding',
                 num_relations=2, embedding_dim=128, expert_dim=64, dropout=0.3, device='cuda'):
        super().__init__()

        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("需要安装 torch_geometric")

        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.expert_dim = expert_dim
        modal_dim = embedding_dim // 4

        # 加载预处理的embedding
        import os
        cat_emb = torch.load(os.path.join(embedding_dir, 'node_cat_embeddings.pt'), map_location='cpu').float()
        num_emb = torch.load(os.path.join(embedding_dir, 'node_num_embeddings.pt'), map_location='cpu').float()
        post_emb = torch.load(os.path.join(embedding_dir, 'node_post_embeddings.pt'), map_location='cpu').float()
        des_emb = torch.load(os.path.join(embedding_dir, 'node_des_embeddings.pt'), map_location='cpu').float()

        self.register_buffer('cat_features', cat_emb.to(self.device))
        self.register_buffer('num_features', num_emb.to(self.device))
        self.register_buffer('post_features', post_emb.to(self.device))
        self.register_buffer('des_features', des_emb.to(self.device))

        print(f"[GraphExpert] 加载特征: cat{cat_emb.shape}, num{num_emb.shape}, "
              f"post{post_emb.shape}, des{des_emb.shape}")

        # 各模态encoder -> D/4
        def make_encoder(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, modal_dim),
                nn.LayerNorm(modal_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout)
            ).to(self.device)

        self.cat_encoder = make_encoder(cat_emb.shape[1])
        self.num_encoder = make_encoder(num_emb.shape[1])
        self.post_encoder = make_encoder(post_emb.shape[1])
        self.des_encoder = make_encoder(des_emb.shape[1])

        # 初始化层
        self.linear_init = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout)
        ).to(self.device)

        # RGCN
        self.rgcn = RGCNConv(embedding_dim, embedding_dim, num_relations=num_relations, num_bases=num_relations)

        # 后处理
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout)
        ).to(self.device)

        # MLP -> expert_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, expert_dim)
        ).to(self.device)

        # 分类头
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, node_indices, edge_index, edge_type):
        # 编码各模态
        h_cat = self.cat_encoder(self.cat_features)
        h_num = self.num_encoder(self.num_features)
        h_post = self.post_encoder(self.post_features)
        h_des = self.des_encoder(self.des_features)

        # 拼接 + 初始化
        x = torch.cat([h_cat, h_num, h_post, h_des], dim=1)
        x = self.linear_init(x)

        # RGCN
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)

        # 提取batch节点
        batch_features = x[node_indices]
        expert_repr = self.mlp(batch_features)
        bot_prob = self.bot_classifier(expert_repr)

        return expert_repr, bot_prob

    def get_expert_repr(self, node_indices, edge_index, edge_type):
        expert_repr, _ = self.forward(node_indices, edge_index, edge_type)
        return expert_repr
