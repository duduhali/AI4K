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
        tag_frame = cv2.resize(frame, (3840, 2160), interpolation=cv2.INTER_CUBIC)
        videoWrite.write(tag_frame)  # 视频写入方法
        (flag, frame) = cap.read()
    videoWrite.release()
    cap.release()

'''每一帧切成四分，分别缩放，然后合并'''
def transform_f4(src_file,tag_file):
    import subprocess
    import glob
    cmd_png = 'ffmpeg -i {0} -vsync 0 {1}%4d.png -y'.format(src_file, tag_file)
    print(cmd_png)  # ffmpeg -i E:/test/one.mp4 -vsync 0 E:/test/one_f4.mp4%4d.png -y
    process_png = subprocess.Popen(cmd_png, shell=True)
    process_png.wait()

    for file in glob.glob('E:/test/one_f4.mp4*.png'):
        scale(file)

    cmd_encoder = 'ffmpeg -r 24000/1001 -i {0}%4d.png  -vcodec libx265 -pix_fmt yuv422p -crf 10 {1}'.format(tag_file,tag_file)
    print(cmd_encoder)
    process_encoder = subprocess.Popen(cmd_encoder, shell=True)
    process_encoder.wait()


from PIL import Image
def scale(file):
    img = Image.open(file)
    pic = Image.new('RGBA', (3840, 2160))
    pic.paste(img.crop((0, 0, 480, 270)).resize((1920, 1080), Image.BICUBIC), (0, 0, 1920, 1080))
    pic.paste(img.crop((480, 0, 960, 270)).resize((1920, 1080), Image.BICUBIC), (1920, 0, 3840, 1080))
    pic.paste(img.crop((0, 270, 480, 540)).resize((1920, 1080), Image.BICUBIC), (0, 1080, 1920, 2160))
    pic.paste(img.crop((480, 270, 960, 540)).resize((1920, 1080), Image.BICUBIC), (1920, 1080, 3840, 2160))
    pic.save(file)

if __name__ == "__main__":
    src_file = 'E:/test/one.mp4'
    tag_file = 'E:/test/one_f4.mp4'
    # transform(src_file,tag_file)
    transform_f4(src_file, tag_file)
#     ffmpeg -i two.mp4 -c:v libx265 s1.mp4   把视频转为 h265编码格式


