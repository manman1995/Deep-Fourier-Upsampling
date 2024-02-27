import torch
Pha = torch.zeros((1,2,6,6))
Pha = torch.tile(Pha, (2, 2))
print(Pha.shape)