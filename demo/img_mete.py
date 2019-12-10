#检测图片的尺寸
import os
from glob import glob
import cv2


data_lr='J:/AI+4K/train_lr'
img_list = glob(os.path.join(data_lr,'*.png'))
print(len(img_list))
size_dict = dict()
for one in img_list:
    img = cv2.imread(one)
    shape = img.shape
    k = '%d_%d'%(shape[1],shape[0])
    v = size_dict.get(k,0)
    size_dict[k] = v+1


print(size_dict)
# 输出结果：{'960_540': 70000}