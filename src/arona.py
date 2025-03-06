import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig
from src.modules import decoder_block
from torch.utils.data import DataLoader, Dataset
import tiktoken
import numpy as np

class ARONA(nn.Module):
    def __init__(self,config=ModelConfig,tie_weights=True):
        super(ARONA, self).__init__()
        self.block_size= config.block_size
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_dim
        )
        self.enc = tiktoken.get_encoding(config.encoding_type)
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]
        self.pos_embedding = nn.Embedding(
            config.block_size,
            config.hidden_dim
        )
        self.layers = nn.Sequential(
           * [decoder_block(config) for _ in range(config.n_layer)]
        )
        self.tokenizer = tiktoken.get_encoding(config.encoding_type)
        self.layernorm_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size,bias=False)
        if tie_weights:
            self.token_embedding.weight=self.lm_head.weight
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, target=None):
        batch_size, seq_len = ids.shape

        # 生成pad_mask [batch, 1, 1, seq_len]
        pad_mask = (ids != ModelConfig.pad_token_id).unsqueeze(1).unsqueeze(2).to(ids.device)

        # Embedding + Positional Encoding
        token_embed = self.token_embedding(ids)  # [batch, seq, dim]
        position_ids = torch.arange(seq_len, device=ids.device).expand(batch_size, seq_len)
        pos_embed = self.pos_embedding(position_ids)  # [batch, seq, dim]
        X = token_embed + pos_embed

        # 逐层传递pad_mask
        for layer in self.layers:
            X = layer(X, mask=pad_mask)  # 需要Decoder Block接收mask

        X = self.layernorm_final(X)
        logits = self.lm_head(X)

        # 计算损失时忽略pad位置
        if target is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=ModelConfig.pad_token_id
            )
        else:
            loss = None

        return logits, loss

    def generate(self, ids, max_new_tokens=100):
        for _ in range(max_new_tokens):
            # 截断至block_size
            if ids.size(1) >= ModelConfig.block_size:
                ids = ids[:, -ModelConfig.block_size:]

            # 生成当前pad_mask
            pad_mask = (ids != ModelConfig.pad_token_id).unsqueeze(1).unsqueeze(2)

            # 前向计算（使用修改后的forward）
            logits, _ = self(ids, mask=pad_mask)

            # 取最后一个token的logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # 终止条件
            if next_token.item() == self.eos_token:
                break

            ids = torch.cat([ids, next_token], dim=-1)

        return ids
    def generate_sentence(self, sentence):
        device = next(self.parameters()).device
        encoded = self.tokenizer.encode(sentence)
        input_len = len(encoded)

        if input_len < self.block_size:
            encoded += [ModelConfig.pad_token_id] * (self.block_size - input_len)
        else:
            encoded = encoded[-self.block_size:]


        input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
        generated_ids = self.generate(input_ids)


        new_ids = [id for id in generated_ids[0].cpu().tolist()
                   if id not in [self.eos_token, ModelConfig.pad_token_id]]

        return self.tokenizer.decode(new_ids)

class AronaDataset(Dataset):
    def __init__(self,config:ModelConfig):
        super(AronaDataset, self).__init__()
        self.block_size = config.block_size
        self.data_max_lines = config.data_max_lines
        self.enc = tiktoken.get_encoding(config.encoding_type)
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]
        data = np.load('data/data.npy')

        data = data[:self.data_max_lines]

        raw_enc_data = []
        for text in data:
            encoded = self.enc.encode(text) 
            encoded.append(self.eos_token)    
            raw_enc_data.append(encoded)   

        self.enc_data = []
        for text in raw_enc_data:
            chunk = text.copy()
            target_length = config.block_size + 1

            if len(chunk) > target_length:
                chunk = chunk[-target_length:]

            elif len(chunk) < target_length:
                pad_needed = target_length - len(chunk)
                chunk += [self.eos_token] * pad_needed

            self.enc_data.append(chunk)

    def __len__(self):
        return len(self.enc_data)

    def __getitem__(self, idx):
        chunk = self.enc_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)