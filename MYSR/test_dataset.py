import utils as util
import argparse
from torch.utils.data import DataLoader
import numpy as np

from dataloderRCAN3D import DatasetLoader


def test(args):
    data_set = DatasetLoader(args.data_lr, args.data_hr, size_w=args.size_w, size_h=args.size_h, scale=args.scale,
                             frame_interval=args.frame_interval,border_mode=args.border_mode,
                             random_reverse=args.random_reverse)
    train_loader = DataLoader(data_set, batch_size=args.batch_size,num_workers=args.workers, shuffle=True,
                              pin_memory=False, drop_last=True)

    for i, train_data in enumerate(train_loader):
        img_lrs = train_data['LRs']
        img_hrs = train_data['HRs']
        print(img_lrs.shape)
        print(img_hrs.shape)


        show(img_lrs,img_hrs)

        if i>=0:
            break

add_mean = util.MeanShift(sign=1)
def show(img_lrs,img_hrs):
    fig = plt.figure()
    hr_data = img_hrs[0, :,:, :, :]
    lr_data = img_lrs[0, :, :, :, :]
    hr_data = hr_data.permute(1, 0, 2, 3)
    lr_data = lr_data.permute(1, 0, 2, 3)
    hr_data = add_mean(hr_data)
    lr_data = add_mean(lr_data)

    for i in range(6):
        sub_img = fig.add_subplot(231+i)
        sub_img.imshow(get_show_data(hr_data[i,:, :, :]))
        sub_img.set_title('hr%d'%i)
    plt.show()

    fig = plt.figure()
    for i in range(6):
        sub_img = fig.add_subplot(231 + i)
        sub_img.imshow(get_show_data(lr_data[i,:, :, :]))
        sub_img.set_title('lr%d'%i)
    plt.show()

def get_show_data(d):
    hr = d.numpy()
    hr = np.clip(hr, 0.0, 255.0).astype(np.uint8)
    hr = np.transpose(hr, (1, 2, 0))
    return hr
from matplotlib import pyplot as plt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size_w', default=256, type=int)
    parser.add_argument('--size_h', default=256, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='J:/5file/train_lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='J:/5file/train_hr')
    # parser.add_argument('--data-lr', type=str, metavar='PATH', default='J:/train_lr')
    # parser.add_argument('--data-hr', type=str, metavar='PATH', default='J:/train_hr')
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--frame_interval', default=[0,1,3,5,7,9] , type=int, nargs='+')
    parser.add_argument('--border_mode', default=True, type=bool)
    parser.add_argument('--random_reverse', default=True, type=bool)

    args = parser.parse_args()
    test(args)