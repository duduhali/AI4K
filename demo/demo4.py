#用缩放图片方法修改视频

import cv2
# pic = cv2.resize(pic, (64, 64), interpolation=cv2.INTER_CUBIC)
# interpolation 选项     所用的插值方法
# INTER_NEAREST    最近邻插值
# INTER_LINEAR     双线性插值（默认设置）
# INTER_AREA   使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于INTER_NEAREST方法。
# INTER_CUBIC  4x4像素邻域的双三次插值
# INTER_LANCZOS4   8x8像素邻域的Lanczos插值


def transform(src_file,tag_file):
    cap = cv2.VideoCapture(src_file)  # 获取一个视频打开句柄
    print(cap.isOpened()) # 判断是否打开

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取宽 高
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(fps, width, height)


    videoWrite = cv2.VideoWriter(tag_file, -1, fps, (3840, 2160))# 1 file 2 编码器 3 帧率 4 size
    (flag, frame) = cap.read()
    while flag:
        tag_frame = cv2.resize(frame, (3840, 2160), interpolation=cv2.INTER_LINEAR)
        videoWrite.write(tag_frame)  # 视频写入方法
        (flag, frame) = cap.read()

    videoWrite.release()
    cap.release()

if __name__ == "__main__":
    src_file = 'E:/SDR_540p/16536366.mp4'
    tag_file = 'E:/16536366.mp4'
    transform(src_file,tag_file)
#     ffmpeg -i two.mp4 -c:v libx265 s1.mp4   把视频转为 h265编码格式

