import json
import dataclasses
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    num_head: int = 2
    decoder_layer: int = 4

