
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from model.SRCNN import SRCNN
from check_utils import calc_psnr

def test(args):
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

    image = Image.open(image_file).convert('RGB')

    '''传入的是低分辨率图'''
    image_width = image.width * scale
    image_height = image.height * scale
    image = image.resize((image_width, image_height), resample=Image.BICUBIC)

    '''传入的是高清图'''
    # image_width = (image.width // scale) * scale
    # image_height = (image.height // scale) * scale
    # image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    # image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    # image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

    image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))
    image, cb, cr = image.convert('YCbCr').split()
    y = np.array(image).astype(np.float32)

    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    out_img_y = preds.clip(0, 255)
    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    out_img.save(image_file.replace('.', '_srcnn_x{}.'.format(scale)))

def work(args):
    cudnn.benchmark = True
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file)["state_dict"].items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()
    all_files = os.listdir(args.image_path)
    with tqdm(total=(len(all_files) - 1)) as t:
        for file in all_files:

            in_file = os.path.join(args.image_path,file)
            out_file = os.path.join(args.out_path,file)
            if os.path.exists(out_file):
                t.update(1)
                continue
            image = Image.open(in_file).convert('RGB')
            image_width = image.width * args.scale
            image_height = image.height * args.scale
            image = image.resize((image_width, image_height), resample=Image.BICUBIC)

            y, cb, cr = image.convert('YCbCr').split()
            y = np.array(y).astype(np.float32)

            y /= 255.
            y = torch.from_numpy(y).to(device)
            y = y.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)
            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
            out_img_y = preds.clip(0, 255)
            out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')
            out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
            out_img.save(out_file)

            t.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    # test(args)
    work(args)

    #python3 test.py --weights-file F:/PycharmProjects/AI4K/SRCNN_Pytorch/weights/x4/best.pth  --image-path J:/abc   --out-path J:/dd