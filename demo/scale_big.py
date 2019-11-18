import cv2
from PIL import Image
import numpy as np


'''保存一张低分辨率图像'''
# cap = cv2.VideoCapture('E:/test/one.mp4')  # 获取一个视频打开句柄
# (flag, frame) = cap.read()
# cv2.imwrite('E:/test/one_L.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0]) #原图
# cap.release()

'''保存一张高清图像'''
# cap = cv2.VideoCapture('E:/test/16536366.mp4')
# (flag, frame) = cap.read()
# cv2.imwrite('E:/test/one_H.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
# cap.release()



'''用OpenCV放大低分辨率图片'''
# img = cv2.imread('E:/test/one_L.png')
# tag_frame = cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_CUBIC) #4x4像素邻域的双三次插值
# cv2.imwrite('E:/test/one_H1.png', tag_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

'''用PIL.Image放大低分辨率图片'''
# img = cv2.imread('E:/test/one_L.png')
# image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #OpenCV转换成PIL.Image格式
# image = image.resize((3840, 2160), Image.BICUBIC) #双三次插值
# image.save('E:/test/one_H2.png')


'''OpenCV 切分成4小分，然后逐个放大，最后合并'''
def scale_big1():
    img = cv2.imread('E:/test/one_L.png')
    print(img.shape)   #(540, 960, 3)        高 和 宽
    #((960 / 2, 540 / 2))  # (480.0, 270.0)
    dst1 = img[0:270,0:480]
    dst2 = img[0:270,480:960]
    dst3 = img[270:540,0:480]
    dst4 = img[270:540,480:960]
    #((3840 / 2, 2160 / 2))  # (1920.0, 1080.0)
    dst1 = cv2.resize(dst1, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    dst2 = cv2.resize(dst2, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    dst3 = cv2.resize(dst3, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    dst4 = cv2.resize(dst4, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    print(dst1.shape,dst2.shape,dst3.shape,dst4.shape)
    # cv2.imwrite('E:/test/H_dst1.png', dst1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # cv2.imwrite('E:/test/H_dst2.png', dst2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # cv2.imwrite('E:/test/H_dst3.png', dst3, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # cv2.imwrite('E:/test/H_dst4.png', dst4, [cv2.IMWRITE_PNG_COMPRESSION, 0])


    part1 = np.concatenate((dst1, dst2), axis=1) #(2160, 1920, 3)
    part2 = np.concatenate((dst3, dst4), axis=1)
    pic = np.concatenate((part1, part2), axis=0)
    print(pic.shape) #(2160, 3840, 3)
    # pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
    cv2.imwrite('E:/test/H_pic.png', pic, [cv2.IMWRITE_PNG_COMPRESSION, 0])


'''PIL.Image 切分成4小分，然后逐个放大，最后合并'''
def scale_big2():
    img = Image.open('E:/test/one_L.png')
    bigWidth, bigHeight = 3840, 2160
    smallWidth, smallHeight = 960, 540
    widthNum,heightNum = 2, 2  # 切成几块

    pic = Image.new('RGBA', (bigWidth, bigHeight))
    for i in range(widthNum):
        for j in range(heightNum):
            dst = img.crop( (i*smallWidth/widthNum, j*smallHeight/heightNum,  (i+1)*smallWidth/widthNum, (j+1)*smallHeight/heightNum) )
            dst = dst.resize((int(bigWidth/widthNum), int(bigHeight/heightNum)), Image.BICUBIC)
            # dst.save( 'E:/test/dst_{0}_{1}.png'.format(i,j) )
            pic.paste(dst, (int(i*bigWidth/widthNum), int(j*bigHeight/heightNum), int((i+1)*bigWidth/widthNum), int((j+1)*bigHeight/heightNum)))

    pic.save('E:/test/H2_pic.png')

# scale_big2()