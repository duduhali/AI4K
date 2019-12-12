#把图片转化为符合提交要求的视频的工具
import argparse
import os, subprocess
import cv2


def transform(img_path,video_path):
    for one in os.listdir(img_path):
        one_dir = os.path.join(img_path,one)
        if len(os.listdir(one_dir))!= 100:
            raise Exception('%s 不等于100帧' %one_dir)

        src = '{}/%3d.png'.format(one_dir)
        dst = '{}/{}.mp4'.format(video_path, one)
        if os.path.exists(dst):
            continue
        cmd_encoder = 'ffmpeg -r 24000/1001 -i ' + src + '  -vcodec libx265 -pix_fmt yuv422p -crf 10 ' + dst
        print(cmd_encoder)
        # ffmpeg -r 24000/1001 -i J:/output/16536366%4d.png  -vcodec libx265 -pix_fmt yuv422p -crf 10 J:/submission/16536366.mp4
        process_encoder = subprocess.Popen(cmd_encoder, shell=True)
        process_encoder.wait()

    print('>>>>>>>>>>>>>>>>>>>>>>> transform OK!!!')

#检测视频是否符合要求
def getVideoSize(video_path):
    import glob
    for file in glob.glob(os.path.join(video_path, '*.mp4')):
        fsize = os.path.getsize(file)
        fsize = fsize / float(1024 * 1024)

        cap = cv2.VideoCapture(file)  # 获取一个视频打开句柄
        if cap.isOpened():  # 判断是否打开
            fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) ) # 获取宽 高
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print('%s: %.2fMB' % (file, fsize), ' fps:{}, width:{}, height:{}'.format(fps, width, height),
                  'frame:{}'.format(frame_count))
            if frame_count != 100:
                raise Exception('帧数不是100', file)
            if width!=3840 or height!=2160:
                raise Exception('宽度或者高度有误', file)

        cap.release()
        if fsize>=60:
            raise Exception('文件大于60M',file)

    print('>>>>>>>>>>>>>>>>>>>>>>> OK!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', default='outputs', type=str)
    parser.add_argument('--video-path', default='RCAN', type=str)
    parser.add_argument('--mode', default=0, type=int,help='1：图片合成视频，2：检测视频，0：合成并检测')
    args = parser.parse_args()

    if args.mode == 0:
        transform(args.img_path, args.video_path) #合成视频
        getVideoSize(args.video_path)  # 检测视频
    elif args.mode == 1:
        transform(args.img_path, args.video_path)  # 合成视频
    elif args.mode == 2:
        getVideoSize(args.video_path)  # 检测视频

    # python3 img2video.py --img-path  J:/output_img --video-path J:/output_mp4



