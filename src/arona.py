import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken
import numpy as np
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
        attn_output = self.MHA(self.layernorm1(X), mask=mask)  
        X = X + attn_output

        ffn_output = self.FFN(self.layernorm2(X))
        X = X + ffn_output

        return X

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


        for layer in self.layers:
            X = layer(X, mask=pad_mask)  # 需要Decoder Block接收mask

        X = self.layernorm_final(X)
        logits = self.lm_head(X)


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
        self.eval()
        with torch.no_grad():  # 禁用梯度，减少内存消耗
            for _ in range(max_new_tokens):
                if ids.size(1) >= ModelConfig.block_size:
                    ids_cond = ids[:, -ModelConfig.block_size:]
                else:
                    ids_cond = ids

                logits, _ = self(ids_cond)
                logits = logits[:, -1, :]  # 取最后一个时间步的logits

                # 检查logits是否包含NaN或inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError("Logits contain NaN or Inf")

                # 实现top_k采样，避免生成EOS和PAD token
                top_k = 40
                temperature = 0.9  # 提高温度增加随机性
                
                # 显式降低EOS和PAD token的概率，鼓励模型生成更多文本
                logits[:, self.eos_token] /= 2.0  # 降低EOS的概率
                logits[:, ModelConfig.pad_token_id] = -float('inf')  # 禁止生成PAD
                
                # 温度缩放
                logits = logits / temperature
                
                # top-k采样
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
                
                probs = F.softmax(logits, dim=-1)
                
                # 打印top-10概率的token，方便调试
                if _ == 0:  # 只对第一个token进行打印
                    top_probs, top_indices = torch.topk(probs, 10)
                    print("Top tokens and their probabilities:")
                    for i, (idx, prob) in enumerate(zip(top_indices[0], top_probs[0])):
                        token = idx.item()
                        try:
                            token_text = self.tokenizer.decode([token])
                            print(f"{i+1}. Token {token} ('{token_text}'): {prob.item():.4f}")
                        except:
                            print(f"{i+1}. Token {token}: {prob.item():.4f}")

                # 检查概率是否有效
                if torch.isnan(probs).any() or (probs < 0).any():
                    raise ValueError("Invalid probabilities")

                next_token = torch.multinomial(probs, num_samples=1)
                
                # 只有当连续生成3个以上EOS或PAD时才停止生成
                # 这是一个简单的防止过早终止的策略
                if next_token.item() == self.eos_token and len(ids[0]) > 10:
                    # 检查前面是否有完整句子，如果有才中断
                    if "。" in self.tokenizer.decode(ids[0][-10:].cpu().tolist()) or \
                       "!" in self.tokenizer.decode(ids[0][-10:].cpu().tolist()) or \
                       "?" in self.tokenizer.decode(ids[0][-10:].cpu().tolist()):
                        break
                
                ids = torch.cat([ids, next_token], dim=-1)
                
                # 每生成10个token打印一次当前输出
                if _ % 10 == 0 and _ > 0:
                    print(f"Generation progress: {self.tokenizer.decode(ids[0].cpu().tolist())}")
                    
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
        
        print(f"Input text: '{sentence}'")
        print(f"Encoded length: {input_len}, tokens: {encoded[:20]}...")
        
        generated_ids = self.generate(input_ids)
        
        # 只截取原始输入后的新生成内容
        new_tokens = generated_ids[0][input_len:].cpu().tolist()
        
        # 过滤掉EOS和PAD tokens
        new_ids = [id for id in new_tokens if id != ModelConfig.pad_token_id]
        
        # 找到第一个EOS token前的所有token
        try:
            eos_index = new_ids.index(self.eos_token)
            new_ids = new_ids[:eos_index]
        except ValueError:
            # 如果没有EOS token，保留所有token
            pass
        
        # 打印生成的原始token
        print(f"Generated {len(new_ids)} new tokens: {new_ids}")
        
        # 解码并返回
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
        data = np.load('/content/drive/MyDrive/TinyArona/data.npy')

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

                chunk += [ModelConfig.pad_token_id] * pad_needed
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
