from data_util import get_iters_att
from util import init_weights
from seq2seq_att import encoder, decoder, seq2seq, attention
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
from seq2seq_att_train import trainer, evaluate
import random
from util import epoch_time
import math
import pandas as pd
import os


input_dim = len(SRC.vocab)
hid_dim = 16
n_layers = 6
n_heads = 8
pf_dim = 128
dropout = 0.1

enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

output_dim = len(TRG.vocab)
hid_dim = 16
n_layers = 6
n_heads = 8
pf_dim = 128
dropout = 0.1

dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

pad_idx = SRC.vocab.stoi['<pad>']

model = Seq2Seq(enc, dec, pad_idx, device).to(device)

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
    
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_iter, criterion)
    
    train_total.append(train_loss)
    valid_total.append(valid_loss)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')