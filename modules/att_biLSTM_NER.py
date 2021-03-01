# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/23 12:07
# @File    : att_biLSTM_NER.py

"""
file description:：

"""
import torch
import torch.nn as nn
from .att_biLSTM import AttBiLSTM
from utils.config import args
import torch.nn.functional as F


class AttBiLSTM4NER(nn.Modules):
    
    def __init__(self, config):
        self.att_biLSTM_layer = AttBiLSTM(config)
        self.classifier = nn.Linear(config.hidden_dim_lstm, config.tag_size)
        self.tag_size = config.tag_size
        self.relation_embedding = nn.Embedding(config.tag_size, config.hidden_dim_lstm)
        self.relation_bias = nn.Parameter(torch.randn(args['batch_size'], config.tag_size, 1))
    
    def forward(self, x):
        attention_output = self.att_biLSTM_layer(x)  # [batch_size, 1, hidden_dim_lstm]
        
        # 下面是论文的做法
        # hidden4cls = F.tanh(attention_output)
        # logits = self.classifier(hidden4cls)
        # logsoft = nn.Softmax(logits)
        
        attention_output = F.tanh(attention_output)
        relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(args['batch_size'], 1)
        relation = self.relation_embedding(relation)
        res = torch.add(torch.bmm(attention_output, relation.transpose(-1,-2)), self.relation_bias)
        logsoft = F.softmax(res, 1)
        
        return logsoft
