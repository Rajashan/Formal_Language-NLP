from data_util import get_iters_trans
from util import init_weights, Opt
from seq2seq_trans import encoder, EncoderLayer, SelfAttention, PositionwiseFeedforward,  decoder, DecoderLayer, seq2seq
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
from seq2seq_trans_train import trainer, evaluate
import random
from util import epoch_time
import math
import pandas as pd
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

train_iter, val_iter, test_iter, train, val, test, SRC, TRG = get_iters_trans([1,0,0],5, 100, 5000, 2000, 1000, 128)

print(f"Number of training examples: {len(train.examples)}")
print(f"Number of validation examples: {len(val.examples)}")
print(f"Number of testing examples: {len(test.examples)}")
print(vars(train.examples[5]))

SRC.build_vocab(train,val)
TRG.build_vocab(train,val)

print(f"Unique tokens in source (SRC) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (TRG) vocabulary: {len(TRG.vocab)}")
input_dim = len(SRC.vocab)
hid_dim = 16
n_layers = 6
n_heads = 8
pf_dim = 128
dropout = 0.1

enc = encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

output_dim = len(TRG.vocab)
hid_dim = 16
n_layers = 6
n_heads = 8
pf_dim = 128
dropout = 0.1

dec = decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

pad_idx = SRC.vocab.stoi['<pad>']

model = seq2seq(enc, dec, pad_idx, device).to(device)

optimizer = Opt(hid_dim, 1, 2000,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

N_EPOCHS = 10
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'transformer-seq2seq.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')
    
train_total = []
valid_total = []
test_total = []

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = trainer(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_iter, criterion)
    
    train_total.append(train_loss)
    valid_total.append(valid_loss)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')