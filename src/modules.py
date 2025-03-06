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

    def forward(self, X, mask=None):
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        batch_size, seq_len, _ = X.shape
        device = X.device

        # --- 动态生成因果掩码 ---
        causal_mask = self.attention_mask[:seq_len, :seq_len].to(device)  # [seq, seq]
        causal_mask = causal_mask.bool()  # 转换为布尔型

        # --- 合并因果掩码和pad_mask ---
        if mask is not None:
            # 调整pad_mask形状 [batch,1,1,seq] => [batch,1,seq,seq]
            pad_mask = mask.expand(-1, -1, seq_len, -1)
            combined_mask = causal_mask.unsqueeze(0) & pad_mask  # [batch,1,seq,seq]
        else:
            combined_mask = causal_mask.unsqueeze(0)  # [1,1,seq,seq]

        # --- 多头拆分 ---
        Q = Q.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)  # [b,h,s,d]
        K = K.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)

        # --- 注意力计算 ---
        attn_scores = Q @ K.transpose(-1, -2) / math.sqrt(Q.size(-1))
        attn_scores = attn_scores.masked_fill(~combined_mask, float('-inf'))  # 关键修改点

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # --- 输出投影 ---
        output = (attn_weights @ V).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        return self.o_proj(output)


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
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config.hidden_dim, eps=config.eps)
        self.MHA = MultiHeadAttetion(config)
        self.FFN = FFN(config)
        self.layernorm2 = nn.LayerNorm(config.hidden_dim, eps=config.eps)

    def forward(self, X, mask=None):
        # 第一层：自注意力 + 残差
        attn_output = self.MHA(self.layernorm1(X), mask=mask)  # 传入mask
        X = X + attn_output

        # 第二层：FFN + 残差
        ffn_output = self.FFN(self.layernorm2(X))
        X = X + ffn_output

        return X
