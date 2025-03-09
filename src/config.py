from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model architecture
    hidden_dim: int = 768         # Embedding size (d_model)
    num_head: int = 4             # Number of attention heads
    n_layer: int = 4              # Number of decoder layers
    ffn_dim: int = 3072           # Feed-forward network dimension (4x hidden_dim)
    dropout: float = 0.1          # Dropout rate
    eps: float = 1e-5             # Layer norm epsilon
    
    # Tokenizer settings
    encoding_type: str = 'gpt2'   # Tokenizer type
    vocab_size: int = 50258       # Vocabulary size for GPT-2 tokenizer
    block_size: int = 1024        # Maximum sequence length
    pad_token_id: int = 50257     # Padding token ID

    # Training settings
    batch_size: int = 4          # Batch size
    num_epochs: int = 20          # Number of training epochs
    learning_rate: float = 5e-5   # Initial learning rate
    min_learning_rate: float = 1e-5  # Minimum learning rate for scheduler
    weight_decay: float = 0.01    # Weight decay for AdamW
    grad_clip: float = 1.0        # Gradient clipping norm
    warmup_steps: int = 2000      # Learning rate warmup steps
    
    # Data settings
    data_max_lines: int = int(1e6)  # Maximum number of lines to load from dataset
    
    # Checkpointing
    checkpoint_interval: int = 5000  # Steps between checkpoints
    save_dir: str = "checkpoints"    # Directory to save checkpoints