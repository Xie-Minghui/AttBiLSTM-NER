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


class AttBiLSTM4NER(nn.Modules):
    
    def __init__(self, config):
        self.att_biLSTM_layer = AttBiLSTM(config)
        self.classifier = nn.Linear(config.hidden_dim_lstm, config.tag_size)
    
    def forward(self, x):
        attention_output = self.att_biLSTM_layer(x)
        hidden4cls = torch.tanh(attention_output)
        logits = self.classifier(hidden4cls)
        logsoft = nn.Softmax(logits)
        
        return logsoft
        
    