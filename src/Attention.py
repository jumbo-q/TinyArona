import torch
import torch.nn as nn
import math
import torch.nn.functional as F

d_model = 512
num_heads = 8
hidden_dim = 256
dropout = 0.2
class self_attention_v1(nn.Module):
    def __init__(self, hiddin_dim):
        super().__init__()
        self.hiddin_dim = hiddin_dim
        self.query_proj = nn.Linear(hiddin_dim, hiddin_dim)
        self.key_proj = nn.Linear(hiddin_dim, hiddin_dim)
        self.value_proj = nn.Linear(hiddin_dim, hiddin_dim)



    # X [batch_size,seq,embedding_dim]
    def forward(self, X):
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        return (
            F.softmax(Q@K.permute(0,2,1)/math.sqrt(self.hiddin_dim), dim=-1)@V
        )


