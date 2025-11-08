import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class DesExpert(nn.Module):
    """
    Description Expert Model
    使用 BERT + MLP 处理 description 信息
    
    输入: description 原始文本
    输出: 
        - 64维 Expert Representation Vector
        - Bot Probability (P(bot|Description Expert))
    """
    def __init__(self, 
                 bert_model_name='bert-base-uncased',
                 hidden_dim=768,
                 expert_dim=64,
                 dropout=0.1):
        """
        Args:
            bert_model_name: BERT 模型名称，默认 'bert-base-uncased'
            hidden_dim: BERT 输出维度，默认 768
            expert_dim: Expert Representation 维度，默认 64
            dropout: Dropout 概率，默认 0.1
        """
        super(DesExpert, self).__init__()
        
        # BERT Encoder (fine-tuned)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        
        # MLP 网络
        # 从 BERT 的 768维 [CLS] token 到 64维 Expert Representation
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
        # 从 64维 Expert Representation 到 1维 bot 概率
        self.bot_classifier = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 bot 概率
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: Tokenized input text, shape: [batch_size, seq_len]
            attention_mask: Attention mask, shape: [batch_size, seq_len]
        
        Returns:
            expert_repr: 64维 Expert Representation Vector, shape: [batch_size, 64]
            bot_prob: Bot Probability, shape: [batch_size, 1]
        """
        # BERT Encoder
        # outputs.last_hidden_state: [batch_size, seq_len, 768]
        # outputs.pooler_output: [batch_size, 768] (CLS token)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_outputs.pooler_output  # [batch_size, 768]
        
        # MLP: 768维 → 64维 Expert Representation
        expert_repr = self.mlp(cls_embedding)  # [batch_size, 64]
        
        # Bot Probability 预测
        bot_prob = self.bot_classifier(expert_repr)  # [batch_size, 1]
        
        return expert_repr, bot_prob
    
    def get_expert_repr(self, input_ids, attention_mask=None):
        """
        只获取 Expert Representation，不计算 bot 概率
        用于 gating network 使用
        
        Args:
            input_ids: Tokenized input text, shape: [batch_size, seq_len]
            attention_mask: Attention mask, shape: [batch_size, seq_len]
        
        Returns:
            expert_repr: 64维 Expert Representation Vector, shape: [batch_size, 64]
        """
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_outputs.pooler_output
        expert_repr = self.mlp(cls_embedding)
        return expert_repr
