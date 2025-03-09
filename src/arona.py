import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from src.config import ModelConfig

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism"""
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, n_heads, len_q, d_k]
            K: [batch_size, n_heads, len_k, d_k]
            V: [batch_size, n_heads, len_v(=len_k), d_v]
            mask: [batch_size, n_heads, len_q, len_k]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1))  # [batch_size, n_heads, len_q, len_k]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        attn_weights = F.softmax(scores, dim=-1)  
        context = torch.matmul(attn_weights, V)   
        return context, attn_weights

class MultiHeadAttetion(nn.Module):
    """Multi-Head Attention module"""
    def __init__(self, config: ModelConfig):
        super(MultiHeadAttetion, self).__init__()
        self.n_heads = config.num_head
        self.head_dim = config.hidden_dim // config.num_head
        self.hidden_dim = config.hidden_dim
        
        self.W_Q = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.W_K = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.W_V = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.attention = ScaledDotProductAttention()
        
        max_seq_len = config.block_size
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, 1, seq_len, seq_len] or None
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        
        if mask is None:
            attention_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        else:
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len].to(x.device)
            
            if mask.dtype == torch.bool:
                mask = mask.float()
            
            attention_mask = causal_mask * mask
        
        context, _ = self.attention(Q, K, V, attention_mask)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(context)
        output = self.dropout(output)
        
        return output

class FFN(nn.Module):
    """Feed-Forward Network module"""
    def __init__(self, config: ModelConfig):
        super(FFN, self).__init__()
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim)
        self.act_layer = nn.GELU()
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        x = self.up_proj(x)
        x = self.act_layer(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    """Transformer decoder block"""
    def __init__(self, config: ModelConfig):
        super(DecoderBlock, self).__init__()
        self.attn_ln = nn.LayerNorm(config.hidden_dim, eps=config.eps)
        self.attn = MultiHeadAttetion(config)
        
        self.ffn_ln = nn.LayerNorm(config.hidden_dim, eps=config.eps)
        self.ffn = FFN(config)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, 1, seq_len, seq_len] or None
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        attn_input = self.attn_ln(x)
        attn_output = self.attn(attn_input, mask)
        x = x + attn_output  
        
        ffn_input = self.ffn_ln(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output  
        
        return x

class ARONA(nn.Module):
    """ARONA language model"""
    def __init__(self, config=ModelConfig()):
        super(ARONA, self).__init__()
        self.config = config
        self.block_size = config.block_size
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.hidden_dim)
        
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        
        self.ln_final = nn.LayerNorm(config.hidden_dim, eps=config.eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        self.token_embedding.weight = self.lm_head.weight
        
        self.tokenizer = tiktoken.get_encoding(config.encoding_type)
        self.eos_token_id = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        """
        Args:
            ids: [batch_size, seq_len] - input token ids
            targets: [batch_size, seq_len] - target token ids (optional)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: scalar tensor (if targets provided)
        """
        batch_size, seq_len = ids.shape
        device = ids.device
        
        token_embeds = self.token_embedding(ids)
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        
        x = token_embeds + pos_embeds
        x = self.emb_dropout(x)
        
        attention_mask = (ids != self.config.pad_token_id).float().unsqueeze(1).unsqueeze(2)
        
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.ln_final(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id
            )
        
        return logits, loss

    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40):
        """Generate text continuation given input prompt ids"""
        self.eval()
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device
        
        ids = prompt_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if ids.size(1) > self.block_size:
                    ids = ids[:, -self.block_size:]
                
                logits, _ = self(ids)
                
                next_token_logits = logits[:, -1, :]
                
                next_token_logits = next_token_logits / temperature
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                
                ids = torch.cat((ids, next_token), dim=1)
                
                if next_token[0].item() == self.eos_token_id:
                    break
        
        return ids

    def generate_sentence(self, sentence):
        device = next(self.parameters()).device
        
        input_ids = torch.tensor([self.tokenizer.encode(sentence)], dtype=torch.long).to(device)
        
        if input_ids.size(1) > self.block_size:
            input_ids = input_ids[:, -self.block_size:]
        
        generated_ids = self.generate(input_ids)
        
        new_tokens = generated_ids[0, input_ids.size(1):].tolist()
        
        if self.eos_token_id in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(self.eos_token_id)]
        
        response = self.tokenizer.decode(new_tokens)
        
        return response

class AronaDataset(Dataset):
    """Dataset for ARONA language model"""
    def __init__(self, config: ModelConfig):
        super(AronaDataset, self).__init__()
        self.config = config
        self.block_size = config.block_size
        self.pad_token_id = config.pad_token_id
        
        self.tokenizer = tiktoken.get_encoding(config.encoding_type)
        self.eos_token_id = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        print("Loading data from data.npy...")
        data = np.load('data.npy', allow_pickle=True)
        if config.data_max_lines > 0:
            data = data[:config.data_max_lines]
        
        print("Preprocessing data...")
        self.examples = []
        for text in data:
            try:
                encoded = self.tokenizer.encode(str(text))
                if len(encoded) < 2: 
                    continue
                    
                encoded.append(self.eos_token_id)  
                
                if len(encoded) <= config.block_size + 1:
                    
                    if len(encoded) < config.block_size + 1:
                        padding_needed = (config.block_size + 1) - len(encoded)
                        chunk = encoded + [config.pad_token_id] * padding_needed
                    else:
                        chunk = encoded
                        
                    self.examples.append((
                        torch.tensor(chunk[:-1], dtype=torch.long),
                        torch.tensor(chunk[1:], dtype=torch.long)
                    ))
                else:

                    for i in range(0, len(encoded) - config.block_size, config.block_size // 2):
                        chunk = encoded[i:i + config.block_size + 1]
                        if len(chunk) < config.block_size + 1:
                            chunk = chunk + [config.pad_token_id] * (config.block_size + 1 - len(chunk))
                            
                        self.examples.append((
                            torch.tensor(chunk[:-1], dtype=torch.long),
                            torch.tensor(chunk[1:], dtype=torch.long)
                        ))
                        
                        if len(self.examples) >= config.data_max_lines: 
                            break
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
                
        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]