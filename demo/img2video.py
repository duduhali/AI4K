
#视频和图片转换
import cv2
import os

#视频分解图片
def getImg(file,img,leve=0):
    # 视频100帧
    cap = cv2.VideoCapture(file)#获取一个视频打开句柄
    isOpened = cap.isOpened() #判断是否打开
    print(isOpened)
    fps = cap.get(cv2.CAP_PROP_FPS)#获取帧率
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)#获取宽 高
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(fps,width,height)
    print(int(width),int(height))

    (flag,frame) = cap.read()
    i = 0
    while flag:
        img_name = '{0}{1}.png'.format(img,i)
        cv2.imwrite(img_name, frame, [cv2.IMWRITE_PNG_COMPRESSION, leve])
        # png 是无损压缩  范围0-9，为9 时压缩比最高
        (flag, frame) = cap.read()
        i += 1
    print(i,flag)
    return

file_1 = 'E:/AI+4K/SDR_540p/10091373.mp4'       #30.0                   (960, 540)
file_2 = 'E:/AI+4K/SDR_4K(Part1)/10091373.mp4' #23.976023976023978      (3840, 2160)  3840/960 = 2160/540 = 4

# file_1 = 'E:/AI+4K/SDR_540p/10099858.mp4'
# file_2 = 'E:/AI+4K/SDR_4K(Part1)/10099858.mp4'

tmp = 'E:/AI+4K/tmp'
tmp2 = 'E:/AI+4K/tmp2'

getImg(file_2,'E:/AI+4K/4K_9/img',9)

# cap = cv2.VideoCapture(file_1)#获取一个视频打开句柄
# print(cap.get(cv2.CAP_PROP_FOURCC)) #编解码的4字-字符代码
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #视频文件中的帧数
# print(cap.get(cv2.CAP_PROP_FORMAT)) #返回对象的格式



#图片合成视频
def getMP4(file_dir, name, fps):
    files = os.listdir(file_dir)
    arr = []
    for one in files:
        file = os.path.join(file_dir, one)
        arr.append(file)

    img = cv2.imread(arr[0]) #拿第一张图片取长宽
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    print(size)

    # fourcc = cv2.VideoWriter_fourcc('A', 'V', 'C', '1') #
    # videoWrite = cv2.VideoWriter(name, fourcc, fps, size)
    videoWrite = cv2.VideoWriter(name, -1, fps, size)
    # 1 file 2 编码器 3 帧率 4 size

    for one in arr:
        img = cv2.imread(one)
        videoWrite.write(img)  # 视频写入方法

# getMP4('E:/AI+4K/tmp',"E:/AI+4K/one.mp4", 30)