# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/25 10:12
# @File    : chinese_utils.py

"""
file description:：
每个字的表示由  字向量  相对实体1的位置向量 相对实体2的位置向量 三者组成

"""
from collections import defaultdict, Iterable
import torch
import copy


def get_relation2id():
    relation2id = {}
    with open('./data/chinese/relation2id.txt', 'r') as f:
        for line in f:
            line = line.rstrip().split()
            relation2id[line[0]] = line[1]
            
    return relation2id


def get_pos_info(num_data):
    relation2id = get_relation2id()
    count_relation = defaultdict(int)
    
    data_sentences, positionE1, positionE2, labels = [], [], [], []
    
    with open('./data/chinese/train.txt', 'r') as f:
        for line in f:
            line = line.rstrip().split()
            if count_relation[line[2]] < num_data:
                count_relation[line[2]] += 1
                
                index1 = line[3].index(line[0])
                index2 = line[3].index(line[1])
                
                position1, position2 = [], []
                sentence = []
                for i, word in enumerate(line[3]):
                    sentence.append(word)
                    position1.append(i-index1)
                    position2.append(i-index2)
                
                data_sentences.append(sentence)
                positionE1.append(position1)
                positionE2.append(position2)
                labels.append(relation2id[line[2]])
    
    return data_sentences, positionE1, positionE2, labels


def flatten(items, ignore_types=(str, bytes)):
    # 将一个多层嵌套的序列展开成一个单层列表
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield flatten(x)
        else:
            yield x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, word2id):
        self.data = copy.deepcopy(data)
        self.word2id = word2id
        self.len_total = len(data['data_sentences'])
        
    def __getitem__(self,index):
        sentence = self.data['data_sentences'][index]
        sentence = self.change_word2id(sentence)
        position1 = self.data['positionE1'][index]
        position2 = self.data['positionE2'][index]
        label = self.data['labels'][index]
        
        data_info = {}
        for key in self.data.keys():
            try:
                data_info[key] = locals()[key]
            except KeyError:
                print('{} cannot be found in locals()'.format(key))
        
        return data_info
    
    def __len__(self):
        return self.len_total
    
    def collate_fn(self, data):
        
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            max_length = max(lengths) if max(lengths) > 0 else 1
            padded_seqs = torch.zeros(len(sequences), max_length)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                seq = torch.Tensor(seq)
                if len(seq) != 0:
                    padded_seqs[i, :end] = seq[:end]
            
            return padded_seqs
        items_info = {}
        for key in data[0].keys():
            items_info[key] = [d[key] for d in data]
        
        sentences = merge(items_info['data_sentences'])
        positionE1 = merge(items_info['positionE1'])
        positionE2 = merge(items_info['positionE2'])
        
        # convert to contiguous and cuda
        sentences = _cuda(sentences.contiguous())
        positionE1 = _cuda(positionE1.contiguous())
        positionE2 = _cuda(positionE2.contiguous())
        labels = _cuda(torch.Tensor(items_info['labels']).contiguous())
        
        data_info = {}
        for key in items_info.keys():
            try:
                data_info[key] = locals[key]
            except KeyError:
                print('{} cannot be found in locals()'.format(key))
        
        return data_info
        
    def change_word2id(self, sentence):
        return [self.word2id[word] for word in sentence]
        
        
def _cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


def get_seqs(batch_size):
    data_sentences, positionE1, positionE2, labels = get_pos_info(1500)
    
    words_all = flatten(data_sentences)
    words_all = set(words_all)
    words_all.add('UNK_TOKEN')
    words_all.add('PAD_TOKEN')
    
    word2id = dict(zip(words_all, range(len(words_all))))
    for i, word in enumerate(words_all):
        word2id[word] = i+2
    word2id['UNK_TOKEN'] = 0
    word2id['PAD_TOKEN'] = 1
    # id2word = dict(zip(range(len(words_all)), words_all))
    
    data = {'data_sentences': data_sentences, 'positionE1': positionE1, 'positionE2': positionE2, 'labels':labels}
    
    dataset = Dataset(data, word2id)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
    )
    
    return  data_loader


