import torch
import torch.nn as nn
from torch.autograd import Variable
x1 = Variable(torch.linspace(0, 81*3-1,81*3).type(torch.FloatTensor)).view(3,3,3,3,3)
print(x1.shape)
print(x1[:, 1, :, :, :].clone().shape)