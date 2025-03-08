from dataclasses import dataclass
@dataclass
class ModelConfig:
    batch_size: int=12
    block_size: int=64
    num_head: int = 16
    n_layer: int = 8
    hidden_dim: int = 1024
    dropout: float=0.1
    eps: float=1e-8
    pad_token_id:int = 50257
    vocab_size: int=50258 
    encoding_type:str = 'gpt2'
    data_max_lines:int= int(4e6)
    pad_token:str = "<|pad|>"

    num_epochs = 20
    learning_rate = 3e-4
    min_learning_rate = 1e-5
    weight_decay = 0.01
    grad_clip = 1.0
    warmup_steps = 1000
    
    t_max = 1000 
    
    checkpoint_interval = 10000 
