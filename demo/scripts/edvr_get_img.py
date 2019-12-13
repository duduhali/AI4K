#为EDVR模型准备数据

import os
import cv2


def getImg(folder_file,sub_folder):
    # 视频100帧
    cap = cv2.VideoCapture(folder_file)
    (flag,frame) = cap.read()
    i = 0
    while flag:
        img_name = os.path.join(sub_folder,'%06d.jpg'%i)
        frame = cv2.resize(frame, (640, 480))
        cv2.imwrite(img_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        # 描述图片质量 范围0-100 有损压缩，为0 时压缩比最高
        (flag, frame) = cap.read()
        i += 1

def edvr(f,f_tag):
    for file in os.listdir(f):
        file_name, extend = os.path.splitext(file)  # '10091373'  '.mp4'
        folder_file = os.path.join(f,file)
        sub_folder = os.path.join(f_tag,file_name)
        if os.path.exists(sub_folder) == False:
            os.makedirs(sub_folder)
            print(sub_folder)

        getImg(folder_file,sub_folder)

gt = 'J:/AI+4K/test/gt'
gt_tag = 'F:/PycharmProjects/AI4K/EDVR-Pytorch/datasets/train/gt'
input = 'J:/AI+4K/test/input'
input_tag = 'F:/PycharmProjects/AI4K/EDVR-Pytorch/datasets/train/input'
edvr(gt,gt_tag)
edvr(input,input_tag)

