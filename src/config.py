from dataclasses import dataclass
@dataclass
class ModelConfig:
    batch_size: int=48
    block_size: int=128
    num_head: int = 2
    n_layer: int = 4
    hidden_dim: int = 1024
    dropout: float=0.1
    eps: float=1e-8
    pad_token_id:int = 50257
    vocab_size: int=50258 
    encoding_type:str = 'gpt2'
    data_max_lines:int= int(4e6)
    pad_token:str = "<|pad|>"
