# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/23 10:36
# @File    : att-bilstm.py

"""
file description:：

"""
import torch
import torch.nn as nn
torch.manual_seed(1)  # 使用相同的初始化种子，保证每次初始化结果一直，便于调试


class Attention(nn.Modules):
    def __init__(self, config):
        self.query = nn.Parameter(torch.randn(config.batch_size, 1, config.hidden_dim_lstm))  # [batch, 1, hidden_dim]
    
    def forward(self, H):
        M = torch.tanh(H)  # H [batch_size, sentence_length, hidden_dim_lstm]
        attention_prob = torch.matmul(M, self.query.transpose(-1, -2))  # [batch_size, sentence_length, 1]
        alpha = nn.Softmax(attention_prob)
        attention_output = torch.matmul(alpha.transpose(-1, -2), H)  # [batch_size, 1, hidden_dim_lstm]
        
        return attention_output
    
        
class AttBiLSTM(nn.Modules):
    def __init__(self, config, embedding_pre):
        super(AttBiLSTM).init()
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim_lstm
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.dropout = nn.Dropout(config.dropout)
        self.pretrained = config.pretrained
        
        assert (self.pretrained is True and embedding_pre is not None) or \
               (self.pretrained is False and embedding_pre is None) , "预训练必须有训练好的embedding_pre"
        # 定义网络层
        # 对于关系抽取，命名实体识别和关系抽取共享编码层
        if self.pretrained:
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
        else:
            self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        
        self.pos1_embedding = nn.Embedding(config.pos_size, config.embedding_dim)
        self.pos2_embedding = nn.Embedding(config.pos_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim_lstm, batch_first=True, bidirectional=True, dropout=config.dropout)
        self.attention_layer = Attention(config)
        
    def forward(self, data_item):
        embeddings = torch.cat((self.word_embedding(data_item['sentences']),
                                self.pos1_embedding(data_item['positionE1']),
                                self.pos2_embedding(data_item['positionE2'])), 1)  # [batch_size, sentence_length, embedding_dim]
        embeddings = self.dropout(embeddings)
        hidden_init = torch.randn(2*self.num_layers, self.batch_size, self.hidden_dim)
        output, h_n = self.gru(embeddings, hidden_init)
        attention_output = self.attention_layer(output)
        #hidden_cls = torch.tanh(attention_output)
        
        return attention_output
        
        
        
        
        
        
        

