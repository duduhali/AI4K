import torch
import torch.nn as nn
from torch.autograd import Variable

#输入维度(seq,batch,feature)
#记忆维度(num_layers*num_directions,batch,hidden)
#num_layers表示RNN单元堆叠的层数，num_directions为1(单向RNN)和2（双向RNN）

#output表示每个时刻网络最后一层的输出，维度是(seq,batch,hidden*num_directions)
# hn是最后时刻所有堆叠的记忆单元的所有输出，维度为(num_layers*num_directions, batch,hidden)
rnn = nn.RNN(2, 4, 1)
# nn.RNN(input_size=20, hidden_size=50, num_layers=2)
input = torch.randn(3, 1, 2)
print(input)
h0 = torch.randn(1, 1, 4)
print(h0)
output, hn = rnn(input, h0)
print(output)
print(output.shape, hn.shape)




# rnn = nn.LSTM(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))
#
#
# rnn = nn.GRU(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# output, hn = rnn(input, h0)
