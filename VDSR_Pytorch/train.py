import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from VDSR_Pytorch.model import VDSR
from VDSR_Pytorch.data_utils import DatasetFromHdf5


def adjust_learning_rate(epoch):
    return lr * (0.1 ** (epoch // step))

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),clip)
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save({'state_dict': model.state_dict(),"epoch": epoch}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main():
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    # train_set = DatasetFromHdf5("data/train.h5")
    # training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batchSize, shuffle=True)

    model = VDSR()
    criterion = nn.MSELoss(size_average=False)  # 返回向量形式
    optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if os.path.isfile(model_pth):
        checkpoint = torch.load(model_pth)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"].state_dict())

    print("===> Training")
    for epoch in range(start_epoch, nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

nEpochs = 50
batchSize = 128
num_workers = 2
momentum = 0.9
weight_decay = 1e-4
lr = 0.1
step = 10
start_epoch = 1
clip = 0.4
model_pth = ''
cuda = torch.cuda.is_available()
if __name__ == "__main__":
    # model = VDSR()
    # print(model.parameters)

    main()


