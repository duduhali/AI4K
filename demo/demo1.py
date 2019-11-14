
#查看视频尺寸
import os
import cv2

file_dir = 'E:/SDR_4K'
#file:10091373.mp4, fps:30.0, width:960.0, height:540.0

# file_dir = 'E:/AI + 4K HDR 赛项/SDR_4K(Part1)'
#file:10091373.mp4, fps:23.976023976023978, width:3840.0, height:2160.0

files = os.listdir(file_dir)
for one in files:
    file = os.path.join(file_dir, one)
    cap = cv2.VideoCapture(file)  # 获取一个视频打开句柄
    if cap.isOpened(): # 判断是否打开
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取宽 高
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('file:{0}, fps:{1}, width:{2}, height:{3}'.format(one,fps,width,height))


