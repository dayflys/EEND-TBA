# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



class MultiHeadSelfAttention(nn.Module):
    """ Multi head "self" attention layer
    """

    def __init__(self, n_units, h=8, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)
        self.d_k = n_units // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout_rate)
        # attention for plot
        self.att = None

    def forward(self, x, batch_size):
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)

        scores = torch.matmul(
                 q.transpose(1, 2), k.permute(0, 2, 3, 1)) / np.sqrt(self.d_k)
        # scores: (B, h, T, T) = (B, h, T, d_k) x (B, h, d_k, T)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        x = torch.matmul(p_att, v.transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, self.h * self.d_k)

        return self.linearO(x)


class PositionwiseFeedForward(nn.Module):
    """ Positionwise feed-forward layer
    """

    def __init__(self, n_units, d_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(n_units, d_units)
        self.linear2 = nn.Linear(d_units, n_units)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """ Positional encoding function
    """

    def __init__(self, n_units, dropout_rate, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        positions = np.arange(0, max_len, dtype='f')[:, None]
        dens = np.exp(
            np.arange(0, n_units, 2, dtype='f') * -(np.log(10000.) / n_units))
        self.enc = np.zeros((max_len, n_units), dtype='f')
        self.enc[:, ::2] = np.sin(positions * dens)
        self.enc[:, 1::2] = np.cos(positions * dens)
        self.scale = np.sqrt(n_units)

    def forward(self, x):
        x = x * self.scale + self.xp.array(self.enc[:, :x.shape[1]])
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, n_layers, n_units,
                 e_units=2048, n_heads=8, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)
        self.pos_enc = PositionalEncoding(n_units, dropout_rate, 5000)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout_rate)
        for i in range(n_layers):
            setattr(self, '{}{:d}'.format("lnorm1_", i),
                    nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format("self_att_", i),
                    MultiHeadSelfAttention(n_units, n_heads, dropout_rate))
            setattr(self, '{}{:d}'.format("lnorm2_", i),
                    nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format("ff_", i),
                    PositionwiseFeedForward(n_units, e_units, dropout_rate))
            
            setattr(self, '{}{:d}'.format("lnorm_out_", i),
                    nn.LayerNorm(n_units))
        #self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x):
        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = x.shape[0] * x.shape[1]
        # e: (BT, F)
        e = self.linear_in(x.reshape(BT_size, -1))
        #e = self.lnorm_in(e)
        aux_list = []
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + self.dropout(s)
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format("ff_", i))(e)
            # residual
            e = e + self.dropout(s)
            if (i+1)%2 == 0:
                aux_list.append(getattr(self, '{}{:d}'.format("lnorm_out_", i))(e))
        # final layer normalization
        # output: (BT, F)
        return aux_list #self.lnorm_out(e)
