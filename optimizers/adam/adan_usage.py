from adan_pytorch import Adan

# mock model

import torch
from torch import nn

model = torch.nn.Sequential(
    nn.Linear(16, 16),
    nn.GELU()
)

# instantiate Adan with model parameters

optim = Adan(
    model.parameters(),
    lr = 1e-3,                  # learning rate (can be much higher than Adam, up to 5-10x)
    betas = (0.02, 0.08, 0.01), # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
    weight_decay = 0.02         # weight decay 0.02 is optimal per author
)

# train

for _ in range(10):
    loss = model(torch.randn(16)).sum()
    loss.backward()
    optim.step()
    optim.zero_grad()