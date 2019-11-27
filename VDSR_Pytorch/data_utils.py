from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, CenterCrop, Resize
import torch
import random

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class DatasetFromFolder(Dataset):
    def __init__(self, image_filenames,target_filenames, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = image_filenames
        self.target_filenames = target_filenames
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        try:
            image, _, _ = Image.open(self.image_filenames[index]).convert('YCbCr').split()
            target, _, _ = Image.open(self.target_filenames[index]).convert('YCbCr').split()
            if self.input_transform:
                image = self.input_transform(image)
            if self.target_transform:
                target = self.target_transform(target)
        except Exception as e:
            random_sum = random.randrange(0, len(self.image_filenames))
            print(index,random_sum,'出现异常',e.__str__())
            return self.__getitem__(random_sum)
        return image, target

    def __len__(self):
        return len(self.image_filenames)

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import glob,os
    import numpy as np
    from torch.utils.data import DataLoader

    LR = 'E:/pngs_cut20/X4'  # (960, 540)
    HR = 'E:/pngs_cut20/gt'  # (3840, 2160)  3840/960 = 2160/540 = 4
    L_files = glob.glob(os.path.join(LR, "*"))
    H_files = glob.glob(os.path.join(HR, "*"))
    L_files = L_files[0:64]
    H_files = H_files[0:64]

    L_files = np.array(L_files)
    H_files = np.array(H_files)

    train_val_scale = 0.8
    train_L = L_files[0:int(len(L_files) * train_val_scale)]
    train_H = H_files[0:int(len(H_files) * train_val_scale)]
    val_L = L_files[int(len(L_files) * train_val_scale):]
    val_H = H_files[int(len(H_files) * train_val_scale):]



    train_set = DatasetFromFolder(train_L, train_H, input_transform=Compose([Resize(( 216, 384), interpolation=Image.BICUBIC),transforms.ToTensor()]),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder(val_L, val_H, input_transform=Compose([Resize(( 216, 384), interpolation=Image.BICUBIC),transforms.ToTensor()]),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=16, shuffle=False)

    for data in train_loader:
        inputs, labels = data
        print(inputs.shape,labels.shape)
        print(labels[0])
        break