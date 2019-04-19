import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np  

class encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim,enc_hid_dim,bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2,dec_hid_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,src, src_len):

        #src = [len(src sentence), batch_size]
        #src_len = [len(src sentence)]

        embedded = self.dropout(self.embedding(src))
        #embedded = [len(src sentence), batch_size, emb_dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
        #outputs = [len(sentence), batch_size, hid_dim * num_directions]
        #hidden = [n_layers * num_directions, batch_size, hid_dim]

        #if bidirectional, hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        #outputs = [len(sentence), batch_size, enc_hid_dim * 2]
        #hidden = [batch_size, dec_hid_dim]

        return outputs, hidden

class attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        #hidden = [batch_size, dec_hid_dim]
        #encoder_outputs = [len(src sentence), batch_size, enc_hid_dim * 2]
        #mask = [batch_size, len(src sentence)]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #hidden = [batch_size, len(src sentence), dec_hid_dim]
        #encoder_outputs = [batch_size, len(src sentence), enc_hid_dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        #energy = [batch_size, dec_hid_dim, len(src sentence)]
        #v = [dec_hid_dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        #v = [batch_size, 1, dec_hid_dim]

        attention = torch.bmm(v, energy).squeeze(1)
        #attention = [batch_size, len(src sentence)]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)

class decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        #input = [batch_size]
        #hidden = [batch_size, dec_hid_dim]
        #encoder_outputs = [len(src sentence), batch_size, enc_hid_dim * 2]
        #mask = [batch_size, len(src sentence)]

        input = input.unsqueeze(0)
        #input = [1, batch_size]

        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch_size, emb_dim]

        a = self.attention(hidden, encoder_outputs, mask)
        #a = [batch_size, len(src sentence)]

        a = a.unsqueeze(1)
        #a = [batch_size, 1, len(src sentence)]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch_size, len(src sentence), enc_hid_dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch_size, 1, enc_hid_dim * 2]

        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch_size, enc_hid_dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        #rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [len(sentence), batch_size, dec_hid_dim * n_directions]
        #hidden = [n_layers * n_directions, batch_size, dec_hid_dim]

        assert (output == hidden).all(), print(output.shape, hidden.shape, output[0,0,:25], hidden[0,0,:25])

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        #output = [batch_size, output_dim]

        return output, hidden.squeeze(0), a.squeeze(1)

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder,pad_idx, sos_idx, eos_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        #src = [len(src sentence), batch_size]
        #src_len = [batch_size]
        #trg = [len(trg sentence), batch_size]

        if trg is None:
            inference = True
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            trg = torch.zeros((100, src.shape[1]), dtype=torch.long).fill_(self.sos_idx).to(src.device)
        else:
            inference = False

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)

        encoder_outputs, hidden = self.encoder(src, src_len)

        output = trg[0,:]
        mask = self.create_mask(src)
        #mask = [batch_size, len(src sentence)]

        for t in range(1, max_len):
            output, hidden, attention = self.decoder(output, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
            if inference and output.item() == self.eos_idx:
                return outputs[:t], attentions[:t]
            
        return outputs, attentions

