from src.arona import AronaDataset
import torch
from torch.utils.data import DataLoader
from src.config import ModelConfig
train_dataset = AronaDataset(ModelConfig)
# split traindataset to train and val
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)
