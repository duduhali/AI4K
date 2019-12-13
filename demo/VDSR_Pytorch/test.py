import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from VDSR_Pytorch.model import VDSR
import matplotlib.pyplot as plt

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def test_one():
    scale = 4
    model_name = "checkpoint/epoch_19_psnr_33.95.pth"
    src_file = 'result/input.bmp'
    dst_file = 'result/input_H.bmp'
    bicubic_file = 'result/input_bicubic.bmp'

    model = VDSR()
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    img = Image.open(src_file)
    bicubic = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    bicubic.save(bicubic_file)
    img = bicubic.convert('YCbCr')

    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
        image = image.cuda()

    out = model(image)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    out_img.save(dst_file)

    # psnr = calc_psnr(y, preds)
    # print('PSNR: {:.2f}'.format(psnr))

    # fig = plt.figure()
    # ax = plt.subplot("131")
    # ax.imshow(img.convert("RGB"))
    # ax.set_title("Input")
    #
    # ax = plt.subplot("132")
    # ax.imshow(bicubic)
    # ax.set_title("Input(bicubic)")
    #
    # ax = plt.subplot("133")
    # ax.imshow(out_img)
    # ax.set_title("Output(vdsr)")
    # plt.show()

def scale_big():
    scale = 4
    model_name = "checkpoint/epoch_19_psnr_33.95.pth"
    src_file = 'J:/test/one_L.png'
    dst_file = 'result/input_H.bmp'
    bicubic_file = 'result/input_bicubic.bmp'

    model = VDSR()
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    img = Image.open(src_file)
    smallWidth, smallHeight = img.width, img.height
    bigWidth, bigHeight = img.width*scale, img.height*scale
    widthNum,heightNum = 3, 3  # 切成几块

    bicubic = img.resize((bigWidth, bigHeight), Image.BICUBIC)
    bicubic.save(bicubic_file)

    pic = Image.new('RGB', (bigWidth, bigHeight))
    for i in range(widthNum):
        for j in range(heightNum):
            print( (i*smallWidth/widthNum, j*smallHeight/heightNum,  (i+1)*smallWidth/widthNum, (j+1)*smallHeight/heightNum) )
            dst = img.crop( (i*smallWidth/widthNum, j*smallHeight/heightNum,  (i+1)*smallWidth/widthNum, (j+1)*smallHeight/heightNum) )

            dst = one(dst,int(bigWidth/widthNum), int(bigHeight/heightNum),model)
            # dst = dst.resize((int(bigWidth/widthNum), int(bigHeight/heightNum)), Image.BICUBIC)

            pic.paste(dst, (int(i*bigWidth/widthNum), int(j*bigHeight/heightNum), int((i+1)*bigWidth/widthNum), int((j+1)*bigHeight/heightNum)))

    pic.save(dst_file)
def one(img,width,height,model):
    img = img.resize((width, height), Image.BICUBIC)
    img = img.convert('YCbCr')

    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
        image = image.cuda()

    out = model(image)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    return out_img


if __name__ == "__main__":
    # test_one()
    scale_big()
    # word()