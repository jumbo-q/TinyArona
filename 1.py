import tiktoken
from src.config import ModelConfig
enc = tiktoken.get_encoding(ModelConfig.encoding_type)
