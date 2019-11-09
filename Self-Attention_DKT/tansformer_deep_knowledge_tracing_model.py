import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np
import math, copy, time


''' This part is going to built a Self-Attention deep knowledge tracing model.'''


def clones(module, N):
    # copy layers
    return nn.ModuleList([copy.deepcopy(module) for n in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) +self.b_2


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key ,value, scale= None, dropout=None):
    # d_k = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1))
    if scale:
        scores = scores * scale
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MuiltHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MuiltHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model//h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = [
        l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
        for l,x in zip(self.linears, (query, key, value))
        ]
        scale = key.size(-1) ** -0.5
        x, self_attn = attention(query, key, value, scale=scale, dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears[-1](x)

class PositionalFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PoitionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PoitionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsequeeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #并没有转置，从二维矩阵转换成只有一列的矩阵
        self.register_buffer('pe', pe)

    def froward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


''' Make a Self-Attention deep knowledge tracing model.'''

def make_model(d_model, d_ff, h, N=6, dropout=0.1):
    c = copy.deepcopy
    attention = MuiltHeadedAttention(h, d_model)
    ff = PositionalFeedForward(d_model, d_ff)
    model = Encoder(EncoderLayer(d_model, c(attention), c(ff), dropout), N)

    for p in model.parameters():
        if p.dim()>1:
            nn.init.kaiming_normal_(p)
    return model
