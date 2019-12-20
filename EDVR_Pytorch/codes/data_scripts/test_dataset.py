import os
from glob import glob
import argparse
from torch.utils.data import DataLoader
import numpy as np

from dataloderREDS import DatasetLoader


def test(args):
    file_name = sorted(os.listdir(args.data_lr))
    lr_list = []
    hr_list = []
    for one in file_name:
        lr_tmp = sorted(glob(os.path.join(args.data_lr, one, '*.png')))
        lr_list.extend(lr_tmp)
        hr_tmp = sorted(glob(os.path.join(args.data_hr, one, '*.png')))
        if len(hr_tmp) != 100:
            print(one)
        hr_list.extend(hr_tmp)

    data_set = DatasetLoader(lr_list, hr_list, size_w=args.size_w, size_h=args.size_h, scale=args.scale,
                             n_frames=args.n_frames,interval_list=args.interval_list,border_mode=args.border_mode,
                             random_reverse=args.random_reverse)
    train_loader = DataLoader(data_set, batch_size=args.batch_size,num_workers=args.workers, shuffle=True,
                              pin_memory=False, drop_last=True)

    for i, train_data in enumerate(train_loader):
        img_lrs = train_data['LRs']
        img_hr = train_data['HR']
        print(img_lrs.shape)
        print(img_hr.shape)


        show(img_lrs,img_hr)

        if i>=0:
            break

def show(img_lrs,img_hr):
    fig = plt.figure()
    sub_img = fig.add_subplot(231)
    sub_img.imshow(get_show_data(img_hr[0, :, :, :]))
    sub_img.set_title('hr')

    sub_img = fig.add_subplot(234)
    sub_img.imshow(get_show_data(img_lrs[0, 0, :, :, :]))
    sub_img.set_title('0')

    sub_img = fig.add_subplot(232)
    sub_img.imshow(get_show_data(img_lrs[0, 1, :, :, :]))
    sub_img.set_title('1')

    sub_img = fig.add_subplot(235)
    sub_img.imshow(get_show_data(img_lrs[0, 2, :, :, :]))
    sub_img.set_title('2')

    sub_img = fig.add_subplot(233)
    sub_img.imshow(get_show_data(img_lrs[0, 3, :, :, :]))
    sub_img.set_title('3')

    sub_img = fig.add_subplot(236)
    sub_img.imshow(get_show_data(img_lrs[0, 4, :, :, :]))
    sub_img.set_title('4')
    plt.show()
def get_show_data(d):
    hr = d.numpy()
    hr *= 255
    hr = np.clip(hr, 0.0, 255.0).astype(np.uint8)
    hr = np.transpose(hr, (1, 2, 0))
    return hr
from matplotlib import pyplot as plt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size_w', default=512, type=int)
    parser.add_argument('--size_h', default=256, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='E:/2file/lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='E:/2file/hr_small')
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--scale', default=1, type=int)
    parser.add_argument('--n_frames', default=5, type=int)
    parser.add_argument('--interval_list', default=[1], type=int, nargs='+')
    parser.add_argument('--border_mode', default=True, type=bool)
    parser.add_argument('--random_reverse', default=True, type=bool)

    args = parser.parse_args()
    test(args)