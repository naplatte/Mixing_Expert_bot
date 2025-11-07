import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class DesExpert(nn.Module):
    def __init__(self, input_dim=768, expert_dim=64, dropout_rate=0.3):
        super(DesExpert, self).__init__()
        
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 冻结BERT的参数，只微调MLP层
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 多层感知机(MLP)用于特征转换
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 专家表示向量输出层
        self.expert_output = nn.Linear(128, expert_dim)
        
        # Bot概率分数输出层（二分类）
        self.bot_output = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x是768维的描述向量
        
        # 由于我们已经有了描述的嵌入向量，这里可以直接传入MLP
        # 如果需要使用BERT进一步处理，可以取消下面注释
        # with torch.no_grad():
        #     bert_output = self.bert(inputs_embeds=x.unsqueeze(1))
        #     x = bert_output.last_hidden_state.squeeze(1)
        
        # 通过MLP进行特征转换
        mlp_output = self.mlp(x)
        mlp_output = self.dropout(mlp_output)
        
        # 生成64维专家表示向量
        expert_vector = self.expert_output(mlp_output)
        expert_vector = F.normalize(expert_vector, p=2, dim=1)  # L2归一化
        
        # 生成bot概率分数
        bot_logits = self.bot_output(mlp_output)
        bot_prob = torch.sigmoid(bot_logits)  # 使用sigmoid激活函数转换为概率
        
        return expert_vector, bot_prob
