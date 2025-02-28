from dataclasses import dataclass
@dataclass
class ModelConfig:
    num_head: int = 2
    decoder_layer: int = 4