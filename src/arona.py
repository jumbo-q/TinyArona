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

    def forward(self,ids,target = None):
        batch_size,seq_len = ids.shape
        token_embedding = self.token_embedding(ids)
        position_ids = torch.arange(seq_len, device=ids.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embedding = self.pos_embedding(
            position_ids
        )

        X = token_embedding + pos_embedding
        X = self.layers(X)
        X = self.layernorm_final(X)
        logits = self.lm_head(X)
        if target is not None:
            batch_size,seq_len,vocab_size = logits.shape
            logits = logits.view(batch_size*seq_len,vocab_size)
            target = target.view(batch_size*seq_len)
            loss = F.cross_entropy(logits, target)
        else:
            loss = None
        return logits,loss
    def generate(self, ids, max_new_tokens=ModelConfig.block_size):
        eos_token = self.eos_token  # 假设EOS token已定义
        for _ in range(max_new_tokens):
            # 截断到block_size
            if ids.size(1) > self.block_size:
                ids = ids[:, -self.block_size:]
            
            logits, _ = self(ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ids_next = torch.multinomial(probs, num_samples=1)
            
            # 检查是否生成EOS
            if ids_next.item() == eos_token:
                ids = torch.cat((ids, ids_next), dim=1)
                break  # 立即终止生成
            
            ids = torch.cat((ids, ids_next), dim=1)
        return ids

    def generate_sentence(self, sentence):
        device = next(self.parameters()).device
        encoded = self.tokenizer.encode(sentence)
        input_len = len(encoded)
        
        # 处理输入至block_size长度
        if input_len < self.block_size:
            encoded += [self.eos_token] * (self.block_size - input_len)
            padded_len = self.block_size
        else:
            encoded = encoded[-self.block_size:]
            padded_len = self.block_size
        
        input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
        generated_ids = self.generate(input_ids)
        
        # 提取新生成的token（block_size之后的部分）
        new_ids = generated_ids[0, padded_len:].cpu().tolist()
        
        # 截断至EOS
        if self.eos_token in new_ids:
            eos_pos = new_ids.index(self.eos_token)
            new_ids = new_ids[:eos_pos]
        
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