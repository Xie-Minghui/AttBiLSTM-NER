# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/23 10:42
# @File    : config.py.py

"""
file description:：

"""


class LSTMConfig:
    def __init__(self,
                 embedding_dim=128,
                 vocab_size=512,
                 pad_token_id=0,
                 hidden_dim_lstm=128,
                 batch_size=8,
                 num_layers=1
                 ):
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.hidden_dim_lstm = hidden_dim_lstm
        self.batch_size = batch_size
        self.num_layers = num_layers
