import torch
import torch.nn as nn
from torch.autograd import Variable

x1 = Variable(torch.linspace(0, 3,4).type(torch.FloatTensor)).view(2,2).unsqueeze_(0)
x2 = Variable(torch.linspace(0, 3,4).type(torch.FloatTensor)).view(2,2).unsqueeze_(0)
x3 = Variable(torch.linspace(0, 3,4).type(torch.FloatTensor)).view(2,2).unsqueeze_(0)
x = torch.cat((x1,x2,x3), 0).unsqueeze_(0)
print(x.size())
print(x)
#torch.Size([1, 3, 2, 2])

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1,bias=True):
        super(MeanShift, self).__init__(3, 3, kernel_size=1,bias=bias)
        std = torch.Tensor(rgb_std)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1) #
        #self.weight.data.shape torch.Size([3, 3, 1, 1]) #这一步不对数据做改变

        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

rgb_range = 255
rgb_mean = (0.4488, 0.4371, 0.4040)
rgb_std = (1.0, 1.0, 1.0)
sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std,bias=True)

output = sub_mean(x)
print(output.shape)
print(output)



# print(torch.Tensor(rgb_mean) / torch.Tensor(rgb_std))
#tensor([0.4488, 0.4371, 0.4040])

# print(rgb_range * torch.Tensor(rgb_mean) / torch.Tensor(rgb_std))
#tensor([114.4440, 111.4605, 103.0200])
x1 = Variable(torch.linspace(0, 3,4).type(torch.FloatTensor)).view(2,2).unsqueeze_(0)
x2 = Variable(torch.linspace(0, 3,4).type(torch.FloatTensor)).view(2,2).unsqueeze_(0)
x3 = Variable(torch.linspace(0, 3,4).type(torch.FloatTensor)).view(2,2).unsqueeze_(0)
x = torch.cat((x1,x2,x3), 0)
print(x.shape)
ssss = (-1*rgb_range * torch.Tensor(rgb_mean) / torch.Tensor(rgb_std))
ssss = ssss.unsqueeze_(1)
ssss = ssss.unsqueeze_(1)
print('ssss.shape',ssss.shape)
ssss = ssss.expand_as(x)
print(ssss.shape)
print(ssss)
print(x+ssss)
# add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
# output = add_mean(x)
# print(output.shape)
# print(output)

