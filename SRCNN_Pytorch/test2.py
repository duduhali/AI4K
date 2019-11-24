"""
Author  : Xu fuyong
Time    : created by 2019/7/17 17:41

"""
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from tqdm import tqdm
from SRCNN_Pytorch.model import SRCNN
from SRCNN_Pytorch.utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr



def test():
    weights_file = 'weights/x4/best.pth'
    # weights_file = 'outputs/x3/best.pth'
    image_file = 'data/one_L_m.png'
    scale = 4
    cudnn.benchmark = True
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()

    image = pil_image.open(image_file).convert('RGB')

    '''传入的是低分辨率图'''
    image_width = image.width * scale
    image_height = image.height * scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

    '''传入的是高清图'''
    # image_width = (image.width // scale) * scale
    # image_height = (image.height // scale) * scale
    # image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    # image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    # image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

    image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(image_file.replace('.', '_srcnn_x{}.'.format(scale)))

def work():
    weights_file = 'weights/x4/best.pth'
    # weights_file = 'outputs/x3/best.pth'
    scale = 4
    image_path = 'J:/img_540p'
    out_path = 'J:/output1'

    cudnn.benchmark = True
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()
    all_files = os.listdir(image_path)
    with tqdm(total=(len(all_files) - 1)) as t:
        for file in all_files:

            in_file = os.path.join(image_path,file)
            out_file = os.path.join(out_path,file)
            if os.path.exists(out_file):
                t.update(1)
                continue
            image = pil_image.open(in_file).convert('RGB')
            image_width = image.width * scale
            image_height = image.height * scale
            image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
            image = np.array(image).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(image)

            y = ycbcr[..., 0]
            y /= 255.
            y = torch.from_numpy(y).to(device)
            y = y.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)
                #将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量

            # psnr = calc_psnr(y, preds)
            # print('PSNR: {:.2f}'.format(psnr))

            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)
            output.save(out_file)

            t.update(1)



if __name__ == '__main__':
    # test()
    work()

