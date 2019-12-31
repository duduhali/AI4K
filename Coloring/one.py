import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
import argparse
import torch.backends.cudnn as cudnn


class DatasetLoader(Dataset):
    def __init__(self, data_list,train=True):
        super(DatasetLoader, self).__init__()
        self.data_list = data_list
        self.train = train
    def __getitem__(self, index):
        data_file = self.data_list[index]
        img = cv2.imread(data_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = np.array(img, dtype=np.float32)
        # augmentation - flip, rotate
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1, :]
            if random.random() < 0.5:
                img = img[::-1, :, :]
            if random.random() < 0.5:
                img = img.transpose(1, 0, 2)
        # HWC to CHW, numpy to tensor
        img = np.transpose(img, (2, 0, 1))
        img_l = img[0,:,:]
        img_l = img_l[np.newaxis,:]
        img_ab = img[1:, :, :]
        img_ab = img_ab/128
        img_l = torch.from_numpy(np.ascontiguousarray(img_l))
        img_ab = torch.from_numpy(np.ascontiguousarray(img_ab))

        return img_l,img_ab
    def __len__(self):
        return len(self.data_list)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=True),
            # nn.ReLU()
        )
        self.conv1 = nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 2, 3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.body(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        # x = self.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv2(x)
        # x = self.relu(x)
        x = self.conv3(x)
        # x = self.tanh(x)
        x = F.interpolate(x, scale_factor=2)
        return x

def main(args):

    data_set = DatasetLoader(glob(args.data_path),False)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=4, shuffle=True,pin_memory=False, drop_last=True)

    print("===> Building model")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # cudnn.benchmark = True

    model = MyModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=0, betas=(0.9, 0.999), eps=1e-08)

    model = model.cuda()
    criterion = criterion.cuda()

    print("===> Training")
    model.train()
    for epoch in range(args.epochs):
        with tqdm(total=(len(data_set) -  len(data_set)%args.batch_size)) as t:
            t.set_description('epoch:{}/{} '.format(epoch, args.epochs - 1))
            for data_x,data_y in train_loader:
                data_x = data_x.cuda()
                data_y = data_y.cuda()

                pred = model(data_x)
                loss = criterion(pred, data_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='loss:%.4f'%loss.item())
                t.update(args.batch_size)

    batch_size = 1
    eval_data_set = DatasetLoader(glob(args.eval_path),train=False)
    eval_train_loader = DataLoader(eval_data_set, batch_size=1, num_workers=4, shuffle=False)
    print("===> Eval")
    model.eval()
    with tqdm(total=(len(eval_data_set) - len(eval_data_set) % batch_size)) as t:
        for i,(data_x, _) in enumerate(eval_train_loader):
            data_x = data_x.cuda()
            pred = model(data_x)

            pred = pred.cpu()
            pred = pred.detach().numpy().astype(np.float32)
            data_x = data_x.cpu()
            data_x = data_x.numpy().astype(np.float32)

            pred = pred * 128

            cur = np.zeros((3,256, 256))
            cur[0,:,:] = data_x[0,:,:,:]
            cur[1:,:,:] = pred[0,:,:,:]
            cur = cur.astype(np.uint8)

            cur = np.transpose(cur, (1, 2, 0))
            cur = cv2.cvtColor(cur, cv2.COLOR_LAB2BGR)

            file_name = "J:/color/result/img_" + str(i) + ".png"
            cv2.imwrite(file_name, cur, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            t.update(batch_size)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.batch_size = 1
    args.epochs = 1000
    args.seed = 123
    args.data_path = 'J:/color/Train/*'
    args.eval_path = 'J:/color/Test/*'
    main(args)