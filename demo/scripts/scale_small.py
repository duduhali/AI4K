# 把图片缩小到1/4
import argparse
from glob import glob
import cv2
import os
from tqdm import tqdm


def small(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)


    file_names = sorted(os.listdir(args.input))
    input_list = []
    for one in file_names:
        input_tmp = sorted(glob(os.path.join(args.input, one, '*.png')))
        input_list.extend(input_tmp)

    with tqdm(total=(len(input_list))) as t:
        for one in input_list:
            file = one.replace('\\', '/')
            arr = file.split('/')
            sub_dir = os.path.join(args.output,arr[-2])
            new_file = os.path.join(sub_dir,arr[-1])

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            img = cv2.imread(file)
            shape = img.shape
            new_image = cv2.resize(img, (shape[1]//4, shape[0]//4))  # 对图片进行缩放
            cv2.imwrite(new_file, new_image)
            t.update(1)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input')
parser.add_argument('--output', type=str, default='output')
args = parser.parse_args()

# args.input = 'J:/5file/train_hr'
# args.output = 'J:/5file/train_hr_small'

small(args)

