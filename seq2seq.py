import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class encoder:
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):

        #src = [len(src sentence), batch_size]

        embedded = self.dropout(self.embedding(input))
        #embedded = [len(src sentence), batch_size, emb_dim]

        output, (hidden, cell) = self.rnn(embedded)

        #outputs = [len(src sentence), batch size, hid_dim * n_directions]
        #hidden = [n_layers * n_directions, batch_size, hid_dim]
        #cell = [n_layers * n_directions, batch_size, hid_dim]
        
        return hidden, cell

class decoder:
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

    #def forward(self, input, hidden, cell):

    
 


