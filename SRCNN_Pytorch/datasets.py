"""
Author  : Xu fuyong
Time    : created by 2019/7/16 19:49

"""
import h5py
import numpy as np
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


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader
    eval_dataset = EvalDataset('J:/AI+4K/pngs_cut20_eval.hdf5')
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    for data in eval_dataloader:
        inputs, labels = data
        print(inputs, labels)