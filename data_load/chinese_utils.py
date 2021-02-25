# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/25 10:12
# @File    : chinese_utils.py

"""
file description:：
每个字的表示由  字向量  相对实体1的位置向量 相对实体2的位置向量 三者组成

"""
from collections import defaultdict, Iterable


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


data_sentences, positionE1, positionE2, labels = get_pos_info(1500)

words_all = flatten(data_sentences)
words_all = set(words_all)
word2id = dict(zip(words_all, range(len(words_all))))
id2word = dict(zip(range(len(words_all)), words_all))

data = {'data_sentences': data_sentences, 'positionE1': positionE1, 'positionE2': positionE2, 'labels':labels}


