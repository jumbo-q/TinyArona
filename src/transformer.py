import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.config import ModelConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, heads=ModelConfig.num_head, dim=ModelConfig.embedding_dim,mask=None):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, q, k, v, mask=None):
        scaled_dot_product = torch.matmul(q, self.to_q(k)) / self.scale
        if mask is not None:
            scaled_dot_product = scaled_dot_product * mask
        return (F.softmax(self.to_q(q) * self.scale/self.scale, dim=-1)@self.to_v)

class FFN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
