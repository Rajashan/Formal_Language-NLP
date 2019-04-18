import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np  

class encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)

        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) for _ in range(n_layers)])

        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        #src = [batch_size, len(src sentence)]
        #src_mask = [batch_size, len(src sentence)]

        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)
        src = self.do((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        #src = [batch_size, len(src sentence), hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch_size, len(src sentence), hid_dim]
        #src_mask = [batch_size, len(src sentence)]
        
        src = self.ln(src + self.do(self.sa(src, src, src, src_mask)))
        
        src = self.ln(src + self.do(self.pf(src)))
        
        return src

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        
        assert hid_dim % n_heads == 0
        
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc = nn.Linear(hid_dim, hid_dim)
        
        self.do = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
        
    def forward(self, query, key, value, mask=None):
        
        bsz = query.shape[0]
        
        #query = key = value [batch_size, len(sentence), hid_dim]
                
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        #Q, K, V = [batch_size, len(sentence), hid_dim]
        
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        
        #Q, K, V = [batch_size, n_heads, len(sentence), hid_dim // n_heads]
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch_size, n_heads, len(sentence), len(sentence)]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = self.do(F.softmax(energy, dim=-1))
        #attention = [batch_size, n_heads, len(sentence), len(sentence)]
        
        x = torch.matmul(attention, V)
        
        #x = [batch_size, n_heads, len(sentence), hid_dim // n_heads]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, sent len, n heads, hid dim // n heads]
        
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        
        #x = [batch_size, len(src sentence), hid_dim]
        
        x = self.fc(x)
        
        #x = [batch_size, len(sentence), hid_dim]
        
        return x

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch_size, len(sentence), hid_dim]
        
        x = x.permute(0, 2, 1)
        
        #x = [batch_size, hid_dim, len(sentence)]
        
        x = self.do(F.relu(self.fc_1(x)))
        
        #x = [batch_size, ff_dim, len(sentence)]
        
        x = self.fc_2(x)
        
        #x = [batch_size, hid_dim, len(sentence)]
        
        x = x.permute(0, 2, 1)
        
        #x = [batch_size, len(sentence), hid_dim]
        
        return x

class decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)
        
        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
                                     for _ in range(n_layers)])
        
        self.fc = nn.Linear(hid_dim, output_dim)
        
        self.do = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, src, trg_mask, src_mask):
        
        #trg = [batch_size, len(trg sentence)]
        #src = [batch_size, len(src sentence)]
        #trg_mask = [batch_size, len(trg sentence)]
        #src_mask = [batch_size, len(src sentence)]
        
        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)
                
        trg = self.do((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg = [batch_size, len(trg sentence), hid_dim]
        
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
            
        return self.fc(trg)

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
        
    def forward(self, trg, src, trg_mask, src_mask):
        
        #trg = [batch_size, len(trg sentence), hid_dim]
        #src = [batch_size, len(src sentence), hid_dim]
        #trg_mask = [batch_size, len(trg sentence)]
        #src_mask = [batch_size, len(src sentence)]
                
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
                
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        
        trg = self.ln(trg + self.do(self.pf(trg)))
        
        return trg

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
        
    def make_masks(self, src, trg):
        
        #src = [batch_size, len(src sentence)]
        #trg = [batch_size, len(trg sentence)]
        
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))
        
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return src_mask, trg_mask
    
    def forward(self, src, trg):
        
        #src = [batch_size, len(src sentence)]
        #trg = [batch_size, len(trg sentence)]
                
        src_mask, trg_mask = self.make_masks(src, trg)
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch_size, len(src sentence), hid_dim]
                
        out = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #out = [batch_size, len(trg sentence), output_dim]
        
        return out

