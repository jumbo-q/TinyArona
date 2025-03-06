import numpy as np

import tiktoken
full_length= 0
enc = tiktoken.get_encoding('gpt2')
eos = enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

print(eos)
