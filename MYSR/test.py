import utils as util
import argparse
from torch.utils.data import DataLoader
import numpy as np

from dataloderB import DatasetLoader


def test(args):
    data_set = DatasetLoader(args.data_lr, args.data_hr, args.size_w, args.size_h, args.scale,args.n_frames, args.interval_list, args.random_reverse)
    train_loader = DataLoader(data_set, batch_size=args.batch_size,num_workers=args.workers, shuffle=True,
                              pin_memory=False, drop_last=True)

    for i, train_data in enumerate(train_loader):
        img_lrs = train_data['LRs']
        img_hr = train_data['HR']
        print(len(img_lrs),img_lrs[0].shape)
        print(img_hr.shape)


        show(img_lrs,img_hr)

        if i>=0:
            break

def show(img_lrs,img_hr):
    fig = plt.figure()

    sub_img = fig.add_subplot(231)
    sub_img.imshow(get_show_data(img_hr[0,:, :, :]))
    sub_img.set_title('hr')
    sub_img = fig.add_subplot(232)
    sub_img.imshow(get_show_data(img_lrs[0][0, :, :, :]))
    sub_img.set_title('0')
    sub_img = fig.add_subplot(233)
    sub_img.imshow(get_show_data(img_lrs[1][0, :, :, :]))
    sub_img.set_title('1')
    sub_img = fig.add_subplot(234)
    sub_img.imshow(get_show_data(img_lrs[2][0, :, :, :]))
    sub_img.set_title('2')
    sub_img = fig.add_subplot(235)
    sub_img.imshow(get_show_data(img_lrs[3][0, :, :, :]))
    sub_img.set_title('3')
    sub_img = fig.add_subplot(236)
    sub_img.imshow(get_show_data(img_lrs[4][0, :, :, :]))
    sub_img.set_title('4')

    plt.show()

def get_show_data(d):
    hr = d.numpy()
    hr = np.clip(hr, 0.0, 255.0).astype(np.uint8)
    hr = np.transpose(hr, (1, 2, 0))
    return hr
from matplotlib import pyplot as plt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size_w', default=128, type=int)
    parser.add_argument('--size_h', default=128, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='J:/5file/train_lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='J:/5file/train_hr')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--n_frames', default=5, type=int)
    parser.add_argument('--interval_list', default=[1,2,3], type=int, nargs='+')
    parser.add_argument('--random_reverse', default=True, type=bool)


    args = parser.parse_args()
    test(args)