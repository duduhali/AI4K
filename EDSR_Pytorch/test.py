import torch
from torch import nn
from demo.visualize import make_dot

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))


x = torch.randn(1,8)
y = model(x)
g = make_dot(y)
g.view()

