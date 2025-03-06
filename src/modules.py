import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.config import ModelConfig
import math


class MultiHeadAttetion(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MultiHeadAttetion, self).__init__()
        self.n_heads = config.num_head
        self.hidden_dim = config.hidden_dim
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.register_buffer(
            'attention_mask',
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size,
                )
            )
        )
        # self.act_layer = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, X):
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        batch,len,_ = X.shape
        state_q = Q.view(batch,len,self.n_heads, -1).transpose(1, 2)
        state_k = K.view(batch,len, self.n_heads, -1).transpose(1, 2)
        state_v = V.view(batch,len, self.n_heads, -1).transpose(1, 2)
        atten_weights = state_q @ state_k.transpose(-1, -2)/math.sqrt(self.hidden_dim)
        atten_weights = atten_weights.masked_fill(self.attention_mask[:] == 0, float('-inf'))
        atten_weights = F.softmax(atten_weights, dim=-1)
        atten_weights = self.dropout(atten_weights)
        out =  atten_weights@state_v
        out = out.view(batch,len,-1)
        out=self.o_proj(out)
        return out
class FFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(FFN, self).__init__()
        self.up_proj = nn.Linear(config.hidden_dim, config.hidden_dim*4)
        self.act_layer = nn.GELU()
        self.down_proj = nn.Linear(config.hidden_dim*4, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim,eps=config.eps)

    def forward(self, X):
        out = self.up_proj(X)
        out = self.act_layer(out)
        out = self.down_proj(out)
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out


class decoder_block(nn.Module):
    def __init__(self, config: ModelConfig):
        super(decoder_block, self).__init__()
        self.layernorm1 = nn.LayerNorm(config.hidden_dim,eps=config.eps)
        self.MHA = MultiHeadAttetion(config)
        self.FFN = FFN(config)
        self.layernorm2 = nn.LayerNorm(config.hidden_dim,eps=config.eps)

    def forward(self, X):
        X=X+self.MHA(self.layernorm1(X))
        X=X+self.FFN(self.layernorm2(X))
        return X
