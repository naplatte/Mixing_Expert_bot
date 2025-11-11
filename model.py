import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# description专家模型
class DesExpert(nn.Module):
    def __init__(self, 
                 bert_model_name='bert-base-uncased',
                 hidden_dim=768,
                 expert_dim=64,
                 dropout=0.1):

        super(DesExpert, self).__init__()
        
        # BERT Encoder (fine-tuned)
        self.bert = BertModel.from_pretrained(bert_model_name)
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
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # BERT 接收到这两个输入后，会对每个 token 做注意力计算，输出对应的上下文表示（每个token的向量表示768维、句子向量768维）
        cls_embedding = bert_outputs.pooler_output  # [batch_size, 768] # 每个输入句子的 整体语义向量（768维），类似之前des处理时的所有token取平均得到句子向量，这里原理类似
        
        # MLP: 768维 → 64维 Expert Representation
        expert_repr = self.mlp(cls_embedding)  # 专家表示，shape:[batch_size, 64]
        
        # Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # shape:[batch_size, 1]
        
        return expert_repr, bot_prob

    # 只获取专家表示，用于 gating network 使用
    def get_expert_repr(self, input_ids, attention_mask=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_outputs.pooler_output
        expert_repr = self.mlp(cls_embedding)
        return expert_repr

# tweets专家模型
class TweetsExpert(nn.Module):
    """
    Tweets Expert Model
    使用 BERT + MLP 处理推文信息，对用户的多条推文做平均聚合
    """
    def __init__(self, 
                 bert_model_name='bert-base-uncased',
                 hidden_dim=768,
                 expert_dim=64,
                 dropout=0.1):
        super(TweetsExpert, self).__init__()
        
        # BERT Encoder (fine-tuned)
        self.bert = BertModel.from_pretrained(bert_model_name)
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
    
    def forward(self, input_ids_list, attention_mask_list=None):
        """
        Args:
            input_ids_list: shape为List[List[Tensor]]
                - 每个元素是一个用户的推文列表list[Tensor]
                - 每个推文是 [seq_len] 的 tensor
                - 例如: [[tweet1_ids], [tweet2_ids], ...] for user 1
            attention_mask_list: mask掩码，1表示有效，0表示填充
        """
        batch_size = len(input_ids_list)
        batch_expert_reprs = [] # 收集整个 batch 中每个用户的“专家表示向量
        
        # 获取设备（从第一个非空用户的推文获取，如果没有则使用默认）
        device = None
        for user_tweets in input_ids_list:
            if len(user_tweets) > 0:
                device = user_tweets[0].device
                break
        if device is None:
            device = next(self.parameters()).device  # 使用模型参数的设备
        
        # 对每个用户的多条推文进行处理
        for user_idx in range(batch_size):
            user_tweets = input_ids_list[user_idx]  # 该用户的所有推文
            user_masks = attention_mask_list[user_idx] if attention_mask_list else None
            
            # 如果用户没有推文（空列表），使用零向量
            if len(user_tweets) == 0:
                expert_repr = torch.zeros(self.mlp[-1].out_features, device=device)
                batch_expert_reprs.append(expert_repr)
                continue
            
            # 对每条推文用 BERT + MLP 处理
            tweet_expert_reprs = []
            for tweet_idx, tweet_ids in enumerate(user_tweets):
                # 确保 input_ids 是 2D: [1, seq_len]
                if tweet_ids.dim() == 1:
                    tweet_ids = tweet_ids.unsqueeze(0)
                
                # 获取 attention mask
                tweet_mask = None
                if user_masks is not None:
                    tweet_mask = user_masks[tweet_idx]
                    if tweet_mask.dim() == 1:
                        tweet_mask = tweet_mask.unsqueeze(0)
                
                # BERT Encoder
                bert_outputs = self.bert(input_ids=tweet_ids, attention_mask=tweet_mask)
                cls_embedding = bert_outputs.pooler_output  # [1, 768]
                
                # MLP: 768维 → 64维 Expert Representation
                tweet_expert_repr = self.mlp(cls_embedding)  # [1, 64]
                tweet_expert_reprs.append(tweet_expert_repr.squeeze(0))  # [64]
            
            # 对所有推文的专家表示做平均聚合
            if len(tweet_expert_reprs) > 0:
                tweet_expert_reprs_tensor = torch.stack(tweet_expert_reprs, dim=0)  # [num_tweets, 64]
                user_expert_repr = torch.mean(tweet_expert_reprs_tensor, dim=0)  # [64] - 平均聚合
            else:
                user_expert_repr = torch.zeros(self.mlp[-1].out_features, device=tweet_expert_reprs[0].device if tweet_expert_reprs else 'cpu')
            
            batch_expert_reprs.append(user_expert_repr)
        
        # Stack all users' expert representations
        expert_repr = torch.stack(batch_expert_reprs, dim=0)  # [batch_size, 64]
        
        # Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]
        
        return expert_repr, bot_prob

    # 只获取专家表示，不计算 bot 概率
    def get_expert_repr(self, input_ids_list, attention_mask_list=None):
        expert_repr, _ = self.forward(input_ids_list, attention_mask_list)
        return expert_repr
    