from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize
import random
import glob,os
import numpy as np
import h5py
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :]/255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

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

#返回文件名列表，size：去路径下的多少个文件，train_val_ratio：训练和验证数据的分割比例
def dealPath(input_path, target_path,size=None, train_val_ratio=None):
    input_files = glob.glob(os.path.join(input_path, "*"))
    target_files = glob.glob(os.path.join(target_path, "*"))

    print(os.path.join(input_path, "*"),len(input_files))
    print(input_files)

    if len(input_files) != len(target_files):
        raise Exception('两边的文件数量不相等')
    if size!=None and size<len(input_files):
        input_files = input_files[0:size]
        target_files = target_files[0:size]

    input_files = np.array(input_files)
    target_files = np.array(target_files)

    if train_val_ratio==None:
        return input_files,target_files
    else:
        train_input = input_files[0:int(len(input_files) * train_val_ratio)]
        train_target = target_files[0:int(len(target_files) * train_val_ratio)]
        val_input = input_files[int(len(input_files) * train_val_ratio):]
        val_target = target_files[int(len(target_files) * train_val_ratio):]
        return train_input,train_target,val_input,val_target

if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    inputs = 'E:test/cut_pngs/X4'  # (960, 540)
    targets = 'E:test/cut_pngs/gt'  # (3840, 2160)  3840/960 = 2160/540 = 4

    # train_input, train_target = dealPath(inputs,targets,size=32)
    # print(len(train_input),len(train_target))
    train_input, train_target, val_input, val_target = dealPath(inputs, targets, train_val_ratio=0.8)
    print(len(train_input), len(train_target),len(val_input),len(val_target))

    train_set = DatasetFromFolder(train_input, train_target,
                                  input_transform=Compose(
                                      [Resize(( 216, 384), interpolation=Image.BICUBIC),
                                       transforms.ToTensor()]),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder(val_input, val_target,
                                input_transform=Compose(
                                    [Resize(( 216, 384), interpolation=Image.BICUBIC),
                                     transforms.ToTensor()]),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=2, drop_last=True,shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=2, drop_last=True,shuffle=True)
    print(len(val_loader))
    for data in train_loader:
        inputs, labels = data
        print(inputs.shape,labels.shape)
        # print(labels[0])
        break