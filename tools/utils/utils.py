import json
import dataclasses
from dataclasses import dataclass, asdict
from torch.utils.data import dataset,DataLoader
from src.config import ModelConfig,TrainConfig
@dataclass
class ModelConfig:
    max_pos = 1800
    d_model = 768  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    CLIP = 1


def make_data(datas):
    train_datas =[]
    for data in datas:
        data=data.strip()
        train_data = [i if i!='\t' else "<sep>" for i in data]+['<sep>']
        train_datas.append(train_data)

    return train_datas

def data_loader(datas):

    return DataLoader(
        dataset=datas,
        batch_size=TrainConfig.batch_size,
        shuffle= True,
        drop_last= True
    )