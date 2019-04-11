# misc
import random
import math
import os
import time
# for data
import numpy as np
import pandas as pd
from nltk.parse.generate import generate
from nltk import CFG
from random import choices, shuffle
from nltk.tokenize import word_tokenize
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
#import spacy
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
# for nn
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

import tensorboardX

def trg2src(text):
    '''[Target to text]
    
    Arguments:
        text {[string]} -- [String of a target language]
    
    Returns:
        output {[string]} -- [String of a source language]
    '''

    output = []
    for sentence in text:
        i = ','.join(sentence)
        b = i.count('{')
        b_ver = i.count('}')
        assert b == b_ver
        
        j = ','.join(sentence)
        a = j.count('(')
        a_ver = j.count(')')
        assert a == a_ver
        output.append(str('a'*a+'b'*b))
        
    return output


def generate_pairs(depth, cfg, N):
    '''
    num_pairs: Integer denoting the number of translation pairs
    depth: integer for thedepth of the parse tree in the CFG
    cfg: chosen grammar, 1, 2 or 3
    '''
    if (cfg == 1):
        grammar = CFG.fromstring("""
        S -> Y  
        Y ->   a Y b | a Y | a  
        a -> '(' ')'  | 
        b -> '{' '}'  |
        """)
    elif cfg == 2:
        grammar = CFG.fromstring("""
        S ->  X | Y  | X Y
        X -> a
        Y ->  b
        a -> '(' a ')'  |  
        b -> '{' b '}'  | 
        """)
    elif cfg == 3:
        grammar = CFG.fromstring("""
        S ->  X 
        X -> a | b
        a -> '(' a ')'  |  
        b -> '{' b '}' | '{' a '}'
        """)
    trg = list(generate(grammar, depth = depth, n = N ))
    trg_list = []
    for sentence in trg:
        k = ''.join(sentence)
        trg_list.append(k)
        
    src_list = trg2src(trg)
    
    if cfg == 1:
        A = list((s + 'A ' for s in src_list))
    elif cfg == 2:
        A = list((s + 'B ' for s in src_list))
    elif cfg == 3:
        A = list((s + 'C ' for s in src_list))
    else:
        None
    
    B = list(( s  for s in trg_list))

    df = pd.concat([pd.Series(A),pd.Series(B)],axis = 1)
    pairs = (df.iloc[:,0]+df.iloc[:,1]).values.tolist()
    return pairs

def get_iters(cfg, depth, num_generated, num_train, num_val, num_test, batch_size):
    '''
    cfg: 3-tuple [a,b,c]
    num_generated:  
    trg: 
    '''
    if np.sum(cfg) == 0:
        print("No language selected")
        return None
    elif np.sum(cfg) == 1:
        language_index = cfg.index(1)+1

        pairs = generate_pairs(depth,language_index,num_generated)

    elif np.sum(cfg) == 2:
        language_index1 = cfg.index(1,0) + 1
        language_index2 = cfg.index(1,1) + 1
        pairs1 = generate_pairs(depth,language_index1,num_generated/2)
        pairs2 = generate_pairs(depth,language_index2,num_generated/2)
        pairs = pairs1 + pairs2

    elif np.sum(cfg) == 3:
        language_index1 = cfg.index(1,0)+1
        language_index2 = cfg.index(1,1)+1
        language_index3 = cfg.index(1,2)+1
        pairs1 = generate_pairs(depth,language_index1,round(num_generated/3))
        pairs2 = generate_pairs(depth,language_index2,round(num_generated/3))
        pairs3 = generate_pairs(depth,language_index3,round(num_generated/3))
        pairs = pairs1 + pairs2 + pairs3


    data = choices(pairs, weights = None, k=num_train + num_val + num_test)

    a,b = map(list, zip(*(s.split(" ") for s in data)))
    a = np.transpose(a)
    b = np.transpose(b)
    a = [" ".join(s) for s in a]
    b = [" ".join(s) for s in b]
    dt = pd.concat([pd.Series(a),pd.Series(b)], axis = 1)
    tokenize_SRC = lambda x: x.split(' ')
    tokenize_TRG = lambda x: x.split(' ')
    SRC = Field(tokenize = tokenize_SRC,init_token='<sos>', eos_token='<eos>', sequential=True, use_vocab=True)
    TRG = Field(tokenize = tokenize_TRG,init_token='<sos>', eos_token='<eos>', sequential=True, use_vocab=True)

    df = pd.DataFrame(dt, columns=["src", "trg"])

    data_fields = [('src', SRC), ('trg', TRG)]

    batch_size = batch_size
    
    if (num_train & num_val):
        val_size = num_val/(num_train+num_val+num_test)
        test_size = (num_test)/(num_val+num_test)
        train, val = train_test_split(dt, test_size= val_size)
        val, test = train_test_split(val, test_size = test_size)

        train.to_csv("train.csv", index=False)
        val.to_csv("val.csv", index=False)
        test.to_csv("test.csv", index=False)

        train,val,test = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', test='test.csv',format='csv', fields=data_fields)

        SRC.build_vocab(train, val)
        TRG.build_vocab(train, val)

        train_iter, val_iter, test_iter = BucketIterator.splits((train,val,test), batch_size=batch_size, device = device, shuffle=True, sort_key=lambda x: len(vars(x)))
        return train_iter, val_iter, test_iter

    else:
        test = dt
        test.to_csv("test.csv", index=False)
        test = TabularDataset.splits(path='./', test='test.csv',format='csv', fields=data_fields)[0]

        SRC.build_vocab(test)
        TRG.build_vocab(test)

        test = TabularDataset.splits(path='./', test='test.csv',format='csv', fields=data_fields)


        test_iter = BucketIterator.splits(test, batch_size=batch_size, device = device, shuffle=True, sort_key=lambda x: len(vars(x)))
        return test_iter
    
    


    


