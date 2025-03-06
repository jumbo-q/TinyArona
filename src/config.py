from dataclasses import dataclass
@dataclass
class ModelConfig:
    batch_size: int=12
    block_size: int=512
    num_head: int = 2
    n_layer: int = 4
    hidden_dim: int = 1024
    dropout: float=0.1
    eps: float=1e-8
    vocab_size: int=50257  
    encoding_type:str = 'gpt2'
    data_max_lines:int=2048
