import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
import argparse
from torchvision import models
from dataloder import DatasetLoader,EvalDatasetLoader

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )#(256,32,32)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.fusion = nn.Conv2d(1256, 256, 1)
        self.fusionBN = nn.BatchNorm2d(256)

        self.decoder1 = nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=True)
        self.decoder1BN = nn.BatchNorm2d(128)

        self.decoder2 = nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=True)
        self.decoder2BN = nn.BatchNorm2d(64)

        self.decoder3 = nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=True)
        self.decoder3BN = nn.BatchNorm2d(32)

        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1, bias=True)
        self.decoder4BN = nn.BatchNorm2d(16)

        self.decoder5 = nn.Conv2d(16, 2, 3, stride=1, padding=1, bias=True)
        self.decoder5BN = nn.BatchNorm2d(2)
    def forward(self, x, fusion):
        x = self.encoder(x)

        # Fusion
        # (1000,)-> (1000,32,32)
        fusion = fusion.unsqueeze_(-1).unsqueeze_(-1)
        fusion = fusion.expand((-1, 1000, 32, 32))
        res = torch.cat([x, fusion], dim=1)
        #(-1,1256,32,32)
        x = self.fusion(res)
        x = self.fusionBN(x)
        x = self.relu(x)
        #decoder
        x = self.decoder1(x)
        x = self.decoder1BN(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.decoder2(x)
        x = self.decoder2BN(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.decoder3(x)
        x = self.decoder3BN(x)
        x = self.relu(x)
        x = self.decoder4(x)
        x = self.decoder4BN(x)
        x = self.relu(x)
        x = self.decoder5(x)
        x = self.decoder5BN(x)
        x = self.tanh(x)
        x = F.interpolate(x, scale_factor=2)
        return x

def train(args):
    data_set = DatasetLoader(glob(args.data_path),args.patch_size)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=4, shuffle=True,pin_memory=False, drop_last=True)

    print("===> Building model")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # cudnn.benchmark = True

    torch.utils.model_zoo.load_url('http://labfile.oss.aliyuncs.com/courses/1073/resnet18-5c106cde.pth')
    resnet = models.resnet18(pretrained=True)
    resnet = resnet.cuda()
    resnet.eval()

    model = MyModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-2, weight_decay=0, betas=(0.9, 0.999), eps=1e-08)

    model = model.cuda()
    criterion = criterion.cuda()

    print("===> Training")
    model.train()
    for epoch in range(args.epochs):
        with tqdm(total=(len(data_set) -  len(data_set)%args.batch_size)) as t:
            t.set_description('epoch:{}/{} '.format(epoch, args.epochs - 1))
            for data_x,data_y,fusion in train_loader:
                fusion = fusion.cuda()
                fusion = resnet(fusion)

                data_x = data_x.cuda()
                data_y = data_y.cuda()

                pred = model(data_x,fusion)
                loss = criterion(pred, data_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='loss:%.4f'%loss.item())
                t.update(args.batch_size)

    torch.save(model.state_dict(), 'model.pth')

def eval(args):
    batch_size = 1
    eval_data_set = EvalDatasetLoader(glob(args.eval_path))
    eval_train_loader = DataLoader(eval_data_set, batch_size=1, num_workers=1, shuffle=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.utils.model_zoo.load_url('http://labfile.oss.aliyuncs.com/courses/1073/resnet18-5c106cde.pth')
    resnet = models.resnet18(pretrained=True)
    resnet = resnet.cuda()
    resnet.eval()

    model = MyModel()
    model.load_state_dict(torch.load('model.pth'))
    model = model.cuda()
    model.eval()
    print("===> Eval")

    with tqdm(total=(len(eval_data_set) - len(eval_data_set) % batch_size)) as t:
        for i, (data_x, _, fusion) in enumerate(eval_train_loader):
            fusion = fusion.cuda()
            fusion = resnet(fusion)

            data_x = data_x.cuda()
            pred = model(data_x, fusion)

            pred = pred.cpu()
            pred = pred.detach().numpy().astype(np.float32)
            data_x = data_x.cpu()
            data_x = data_x.numpy().astype(np.float32)

            pred = pred * 128
            cur = np.zeros((3, 256, 256))
            cur[0, :, :] = data_x[0, :, :, :]
            cur[1:, :, :] = pred[0, :, :, :]
            cur = cur.astype(np.uint8)

            cur = np.transpose(cur, (1, 2, 0))
            cur = cv2.cvtColor(cur, cv2.COLOR_LAB2BGR)

            file_name = "J:/color/result/img_" + str(i) + ".png"
            cv2.imwrite(file_name, cur, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            t.update(batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.batch_size = 32
    args.epochs = 100
    args.patch_size = 256
    args.seed = 123
    args.data_path = 'E:/AI/model_search/new/*'
    args.eval_path = 'J:/color/Test/*'

    train(args)
    eval(args)