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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

train_iter, val_iter, test_iter, train, val, test, SRC, TRG = get_iters_att([1,0,0],10, 100000, 5000, 2000, 1000, 128)

print(f"Number of training examples: {len(train.examples)}")
print(f"Number of validation examples: {len(val.examples)}")
print(f"Number of testing examples: {len(test.examples)}")
print(vars(train.examples[5]))

SRC.build_vocab(train,val)
TRG.build_vocab(train,val)

print(f"Unique tokens in source (SRC) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (TRG) vocabulary: {len(TRG.vocab)}")

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
PAD_IDX = SRC.vocab.stoi['<pad>']
SOS_IDX = TRG.vocab.stoi['<sos>']
EOS_IDX = TRG.vocab.stoi['<eos>']

attn = attention(ENC_HID_DIM, DEC_HID_DIM)
enc = encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = seq2seq(enc, dec, PAD_IDX, SOS_IDX, EOS_IDX, device).to(device)
print(model)
optimizer = optim.Adam(model.parameters())

pad_idx = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

N_EPOCHS = 10
CLIP = 10
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'seq2seq_att_model.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')
    
train_total = []
valid_total = []
test_total = []

for epoch in range(N_EPOCHS):
    
    train_loss = trainer(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_iter, criterion)
    
    train_total.append(train_loss)
    valid_total.append(valid_loss)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')

epoch_axis = np.arange(len(train_total))

f, (ax1, ax2) = plt.subplots(1,2, sharex = True, sharey = True, figsize=(15,5))

ax1.plot(epoch_axis, np.asarray(train_total).squeeze(), 'r', epoch_axis, np.asarray(valid_total).squeeze(), 'b')
ax2.plot(epoch_axis, np.asarray(np.exp(train_total)).squeeze(), 'r', epoch_axis, np.asarray(np.exp(valid_total)).squeeze(), 'b')

ax1.legend(['Train Loss','Validation Loss'])
ax2.legend(['Train PPL','Validation PPL'])