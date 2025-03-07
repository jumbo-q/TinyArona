import tiktoken
from src.config import ModelConfig
enc = tiktoken.get_encoding(ModelConfig.encoding_type)
print("你好呀",enc.encode("你好呀"))
print("你好呀",enc.decode([19526, 254, 25001, 121, 37772, 222]))