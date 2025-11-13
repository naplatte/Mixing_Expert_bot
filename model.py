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

# 门控网络:输入所有专家的表示向量（拼接后），输出每个专家的权重
class GatingNetwork(nn.Module):
    def __init__(self, num_experts, expert_dim=64, hidden_dim=128):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts # 专家数量
        self.fc1 = nn.Linear(num_experts * expert_dim, hidden_dim) # 专家数量 * 专家表示维度，即num*64的矩阵->128d的向量
        self.fc2 = nn.Linear(hidden_dim, num_experts) # 128d->专家数量d的向量 比如5个专家就是[x,x,x,x,x]，每个x就是该专家的打分值（之后经过Softmax转为权重）

    def forward(self, expert_reprs, active_mask=None): # 输入：
        """
        expert_reprs: Tensor [batch_size, num_experts, expert_dim] - 每个用户的表示为[num_experts, expert_dim]即num_experts*expert_dim的矩阵
        active_mask:  Tensor [batch_size, num_experts], 1 表示该专家激活，0 表示缺失 表示对于该用户，专家激活/缺失的情况
        """
        # 拼接所有专家的表示向量
        x = expert_reprs.reshape(expert_reprs.size(0), -1)   # 将专家表示矩阵展平成一个长向量，方便送入MLP

        # 两层 MLP
        h = torch.tanh(self.fc1(x)) # 非线性特征提取
        logits = self.fc2(h) # 输出每个专家的打分值 [batch_size, num_experts]

        # 对缺失专家屏蔽（防止影响softmax）
        if active_mask is not None:
            logits = logits.masked_fill(active_mask == 0, -1e9) # 将mask为0（专家未激活）的logits对应位置设为-1e9(fc1和fc2对所有输入向量都做计算，不管某个专家是否“激活”，它都会在logits输出一个对应位置的值)

        # Softmax 得到权重（权重之和为 1）
        weights = F.softmax(logits, dim=1)
        return weights   # [batch_size, num_experts]


# 专家聚合:负责将多个专家的输出（64d向量 + 概率）通过门控网络加权融合。
class ExpertGatedAggregator(nn.Module):
    def __init__(self, num_experts, expert_dim=64, hidden_dim=128):
        super(ExpertGatedAggregator, self).__init__()
        self.num_experts = num_experts
        self.gating = GatingNetwork(num_experts, expert_dim, hidden_dim)

    def forward(self, expert_reprs_list, expert_bot_probs_list, active_mask=None):
        """
        expert_reprs_list: 长度为专家数量的列表，每个元素是[batch_size, expert_dim]，表示每个专家在每个batch中的表示向量
        expert_bot_probs_list: 长度为 num_experts 的列表，每个元素是 [batch_size, 1]
        active_mask: [batch_size, num_experts]，指示哪些专家有效
        """
        # 堆叠为张量
        expert_reprs = torch.stack(expert_reprs_list, dim=1)  # [batch_size, num_experts, expert_dim]
        expert_probs = torch.stack([p.squeeze(-1) for p in expert_bot_probs_list], dim=1)  # [batch_size, num_experts]

        # 获取门控网络输出的专家权重
        weights = self.gating(expert_reprs, active_mask)      # [batch_size, num_experts]

        # 根据权重对各专家输出概率加权求和
        final_prob = (weights * expert_probs).sum(dim=1, keepdim=True)  # [batch_size, 1]

        return final_prob, weights