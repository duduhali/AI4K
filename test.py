import torch
import torch.nn as nn
from torch.autograd import Variable

x = Variable(torch.linspace(0, 47,48).type(torch.FloatTensor)).view(1,4,3,2,2)
print(x.size())

class MeanShift(nn.Conv3d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), sign=-1):
        super(MeanShift, self).__init__(4, 4, kernel_size=1)

        self.weight.data = torch.eye(4).view(4, 4, 1, 1,1)
        #self.weight.data.shape torch.Size([3, 3, 1, 1]) #这一步不对数据做改变

        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        for p in self.parameters():
            p.requires_grad = False

rgb_range = 255
rgb_mean = (0,0.4488, 0.4371, 0.4040)
sub_mean = MeanShift(rgb_range, rgb_mean)

output = sub_mean(x)
print(output.shape,output)


