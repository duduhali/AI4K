#把图片转化为符合提交要求的视频的工具
import argparse
import os, subprocess
import cv2
import sys
from multiprocessing import Pool


class ProgressBar(object):
    def __init__(self,task_num=0):
        self.task_num = task_num
        self.completed = 0
    def update(self, msg):
        self.completed += 1
        sys.stdout.write('{}/{}    {}'.format(self.completed, self.task_num, msg))
        sys.stdout.flush()

def makeMP4(img_path, video_path, one):
    one_dir = os.path.join(img_path, one)
    if len(os.listdir(one_dir)) != 100:
        raise Exception('%s 不等于100帧' % one_dir)

    src = '{}/%3d.png'.format(one_dir)
    dst = '{}/{}.mp4'.format(video_path, one)
    if os.path.exists(dst):
        return dst,'exist'
    cmd_encoder = 'ffmpeg -r 24000/1001 -i ' + src + '  -vcodec libx265 -pix_fmt yuv422p -crf 10 ' + dst
    # print(cmd_encoder)
    # # ffmpeg -r 24000/1001 -i J:/output/16536366%4d.png  -vcodec libx265 -pix_fmt yuv422p -crf 10 J:/submission/16536366.mp4
    os.system(cmd_encoder)
    # process_encoder = subprocess.Popen(cmd_encoder, shell=True)
    # process_encoder.wait()
    return dst,'ok'

def transform(args):
    if not os.path.exists(args.video_path):
        os.makedirs(args.video_path)
    sub_folder_list = os.listdir(args.img_path)

    pbar = ProgressBar(len(sub_folder_list))
    def mycallback(param):
        dst = param[0]
        state = param[1]
        pbar.update('{}:{}'.format(state, dst))
    pool = Pool(args.n_thread)
    for one in sub_folder_list:
        pool.apply_async(makeMP4, args=(args.img_path, args.video_path, one), callback=mycallback)
    pool.close()
    pool.join()

    print('>>>>>>>>>>>>>>>>>>>>>>> transform OK!!!')

#检测视频是否符合要求
def getVideoSize(args):
    import glob
    for file in glob.glob(os.path.join(args.video_path, '*.mp4')):
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
    parser.add_argument('--n_thread', default=1, type=int)
    parser.add_argument('--mode', default=0, type=int,help='1：图片合成视频，2：检测视频，0：合成并检测')
    args = parser.parse_args()

    if args.mode == 0:
        transform(args) #合成视频
        getVideoSize(args)  # 检测视频
    elif args.mode == 1:
        transform(args)  # 合成视频
    elif args.mode == 2:
        getVideoSize(args.video_path)  # 检测视频

    # python3 img2video.py --img-path  ./output_img --video-path ./output_mp4 --n_thread 4
    # cd output_mp4
    # zip -r rcan28.zip ./*
    # cd ..
    # ./naic_submit -token 2fef0f957ee8ed2b -file ./output_mp4/*.zip



