import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

def make_data(datas):
    train_datas =[]
    for data in datas:
        data=data.strip()
        train_data = [i if i!='\t' else "<sep>" for i in data]+['<sep>']
        train_datas.append(train_data)

    return train_datas

class SingleHeadAttehtion(nn.modules):
    def __init__(self,config):
        super().__init__()

class MultiHeadAttention(nn.modules):
    def __init__(self):
        super().__inti__()


class Arona(nn.modules):
    def __init__(self):
        super().__init__()
