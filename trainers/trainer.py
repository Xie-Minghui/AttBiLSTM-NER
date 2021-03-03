# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/23 15:53
# @File    : trainer.py

"""
file description:：

"""
from modules.att_biLSTM_NER import AttBiLSTM4NER

import torch
import torch.nn as nn
from tqdm import tqdm
from utils.config import *


if torch.cuda.is_available():
    USE_CUDA = True


class Trainer:
    def __init__(self,
                 model,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None
                 ):
        
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args['lr'])

        # 学习率调控
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optomizer, factor=0.5, patience=3, min_lr=1e-3,
                                                                   verbose=True)

        self.criterion = nn.CrossEntropyLoss()

        if USE_CUDA:
            self.model = self.model.cuda()
        
    def train(self):
        print('STARTING TRAIN...')
        for epoch in range(args['epochs']):
            print("Epoch: {}".format(epoch))
            pbar = tqdm(enumerate(self.train_dataset), total=len(self.train_dataset))
            
            for i, data in pbar:
                self.train_batch(data)
                pbar.set_description('TRAIN LOSS: {}'.format(self.loss))
            
            if (epoch+1) % args['eval_period'] == 0:
                loss = self.evaluate(self.dev_dataset)
                self.model.scheduler.step(loss)  # 调控学习率这个超参数，还可以调整掐超参数吗？因为验证集的作用就是调超参

    def train_batch(self, data):
        
        #zero gradients of all optomizers
        self.model.zero_grad()
        
        logsoft = self.model.forward(data['features'])
        
        self.loss = self.criterion(logsoft, data['label'])
        self.loss.backward()
        self.optomizer.step()
    
    def evaluate(self, dataset):
        print('STARTING EVALUATION...')
        self.model.train(False)
        pbar = tqdm(enumerate(dataset), total=len(dataset))
        
        loss_total = 0
        for i, data in pbar:
            self.train_batch(data)
            # pbar.set_description('EVAL LOSS: {}'.format(self.loss))
            loss_total += self.loss
        self.model.train(True)
        
        return loss_total
