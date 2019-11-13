
# 把生成花瓣案例的花瓣数据集缩小到1/4像素(供搭建超分模型使用)
import glob
from scipy import misc
import os
images_path = "E:/data/images/*"
small_images_dir = "E:/data/small_images"
for image_file in glob.glob(images_path):
    # print(image_file)                     #E:/data/images\image_00001.jpg
    file_name = image_file.split('\\')[-1] #image_00001.jpg
    image = misc.imread(image_file)
    shape = image.shape #(64, 64, 3)
    new_image = misc.imresize(image,(16,16)) #对图片进行缩放
    new_image_file = os.path.join(small_images_dir,file_name)
    misc.imsave(new_image_file, new_image)

