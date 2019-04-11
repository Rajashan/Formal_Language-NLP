from data_util import get_lang, get_iters
from util import init_weights
from seq2seq import encoder, decoder, seq2seq
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
from seq2seq_train import trainer, evaluate
import random
from util import epoch_time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)
SRC, TRG, train, val, test = get_lang([1,0,0],10, 100, 5000, 2000, 0, 128)
train_iter, val_iter, test_iter = get_iters([1,0,0],10, 100, 5000, 2000, 0, 128)

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
ENC_EMB_DIM = 10
DEC_EMB_DIM = 10
HID_DIM = 64
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = seq2seq(enc, dec, device)
print(model)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

PAD_IDX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

N_EPOCHS = 10
CLIP = 1

# for saving model with best val loss
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = trainer(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'seq2seq-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
