from data_util import get_lang, get_iters
from util import init_weights
from seq2seq import encoder, decoder, seq2seq
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, val_iter, trest_iter = get_iters([1,0,0],10, 100, 5000, 2000, 0, 128)
SRC, TRG, train, val, test = get_lang([1,0,0],10, 100, 5000, 2000, 0, 128)

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



