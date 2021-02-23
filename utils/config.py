# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/23 10:42
# @File    : config.py.py

"""
file description:：

"""
import argparse


class LSTMConfig:
    def __init__(self,
                 embedding_dim=128,
                 vocab_size=512,
                 pad_token_id=0,
                 hidden_dim_lstm=128,
                 batch_size=8,
                 num_layers=1,  # lstm的层数
                 dropout=0.1,
                 tag_size=10  # 标记的种类
                 ):
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.hidden_dim_lstm = hidden_dim_lstm
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.tag_size = tag_size


def args_parser():
    parser = argparse.ArgumentParser(description='AttBiLSTM的配置')
    parser.add_argument('-e', '--epochs', help='Epochs Number', type=int)
    parser.add_argument('-ep', '--eval_period', help='Eval Period', type=int)
    args = vars(parser.parse_args())
    
    return args


args = args_parser()
