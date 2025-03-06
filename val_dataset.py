import torch
from torch.utils.data import DataLoader
from src.config import ModelConfig
from src.arona import ARONA

model = ARONA(ModelConfig)
test_ids = torch.tensor([
    [1, 2, ModelConfig.pad_token_id], 
    [3, ModelConfig.pad_token_id, ModelConfig.pad_token_id]
])

# 前向计算
logits, _ = model(test_ids)

# 检查输出中pad位置的logits是否为随机值
# 预期：pad位置（索引2和索引1/2）的预测概率应接近均匀分布
print("Pad位置的logits方差：", logits[:, 2, :].var(dim=-1))