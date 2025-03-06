import numpy as np


data = np.load(
    '../data/data.npy',
)

full_length=0
for i in data:
    full_length += len(i)
print(data.shape)
print(full_length)