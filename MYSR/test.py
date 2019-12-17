import torch
import torch.nn as nn
from torch.autograd import Variable

lr_file = '/aaa/bbb/1050345\\050.png'
lr_file = lr_file.replace('\\','/')

file_name = lr_file.split('/')[-1]
name = file_name.split('.')[0]
print(name)

x = int(name)
b = x-1 if x!=1 else x+1
f = x+1 if x!=100 else x-1

b_file = lr_file.replace(file_name,'%03d.png'%b)
f_file = lr_file.replace(file_name,'%03d.png'%f)

print(b_file)
print(f_file)


