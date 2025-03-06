import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig
from src.modules import decoder_block

class ARONA(nn.Module):
    def __init__(self,config=ModelConfig,tie_weights=True):
        super(ARONA, self).__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_dim
        )
        self.pos_embedding = nn.Embedding(
            config.block_size,
            config.hidden_dim
        )
        self.layers = nn.Sequential(
           * [decoder_block(config) for _ in range(config.n_layer)]
        )
        self.layernorm_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size,bias=False)
        if tie_weights:
            self.token_embedding.weight=self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx,target = None):
        batch_size,seq_len = idx.shape
        token_embedding = self.token_embedding(idx)
        pos_embedding = self.pos_embedding(idx)
        X = token_embedding + pos_embedding
        X = self.layers(X)
        X = self.layernorm_final(X)
        logits = self.lm_head(X)
        if target is not None:
            batch_size,seq_len,vocab_size = logits.shape
            logits = logits.view(batch_size*seq_len,vocab_size)
            target = target.view(batch_size*seq_len)
            loss = F.cross_entropy(X, target)
        else:
            loss = None
        return logits,loss