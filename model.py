import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

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
    