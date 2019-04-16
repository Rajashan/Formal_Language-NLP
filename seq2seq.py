import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np  

class encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):

        #src = [len(src sentence), batch_size]

        embedded = self.dropout(self.embedding(src))
        #embedded = [len(src sentence), batch_size, emb_dim]

        output, (hidden, cell) = self.rnn(embedded)

        #outputs = [len(src sentence), batch size, hid_dim * n_directions]
        #hidden = [n_layers * n_directions, batch_size, hid_dim]
        #cell = [n_layers * n_directions, batch_size, hid_dim]
        
        return hidden, cell


class decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim,emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        #input = [batch_size]
        #hidden = [n_layers * n_directions, batch_size, hid_dim]
        #cell = [n_layers * n_directions, batch_size, hid_dim]

        input = input.unsqueeze(0)

        #input = [1, batch_size]

        embedded = self.dropout(self.embedding(input))

        #embedded = [1, batch_size, emb_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        #output = [len(sentence), batch_size, hid_dim * n_directions]
        #hidden = [n_layers * n_directions, batch_size, hid_dim]
        #cell = [n_layers * n_directions, batch_size, hid_dim]

        prediction = self.out(output.squeeze(0))
        #prediction = [batch size, output dim]

        return prediction, hidden, cell

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal."
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers."

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [len(src sentence), batch_size]
        #trg = [len(trg sentence), batch_size]

        batch_size = trg.shape[1]
        max_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
            
        hidden, cell = self.encoder(src)

        input = trg[0,:]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top = output.max(1)[1]
            input = (trg[t] if teacher_force else top)

        return outputs







    
 


