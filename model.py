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

    # 只获取 Expert Representation，不计算 bot 概率，用于 gating network 使用
    def get_expert_repr(self, input_ids, attention_mask=None):
        """
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
