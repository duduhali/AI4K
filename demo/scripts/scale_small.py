
# 把生成花瓣案例的花瓣数据集缩小到1/4像素(供搭建超分模型使用)
import glob
import cv2
import os
images_path = "E:/ai_data/flower/images/*"
small_images_dir = "E:/ai_data/flower/small_images"
for image_file in glob.glob(images_path):
    # print(image_file)                     #E:/data/images\image_00001.jpg
    file_name = image_file.split('\\')[-1] #image_00001.jpg
    img = cv2.imread(image_file)
    # shape = img.shape #(64, 64, 3)
    # print(shape)

    new_image = cv2.resize(img,(16,16)) #对图片进行缩放
    new_file = os.path.join(small_images_dir,file_name)
    cv2.imwrite(new_file, new_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # print(new_image.shape)

