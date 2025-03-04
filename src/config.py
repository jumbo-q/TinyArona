from dataclasses import dataclass
@dataclass
class ModelConfig:
    num_head: int = 2
    decoder_layer: int = 4
    embedding_dim:int = 256
@dataclass
class TrainConfig:
    batch_size:int=2