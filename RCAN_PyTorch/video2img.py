# 图片单独存放，视频名作为文件夹
import glob,os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mp4-path', type=str)
parser.add_argument('--img-path', type=str)
args = parser.parse_args()

# args.mp4_path = 'E:/2file/hr_mp4'
# args.img_path = 'E:/2file/hr'

if not os.path.exists(args.img_path):
    os.mkdir(args.img_path)
new_name = args.mp4_path + '/*.mp4'
video_lists = sorted(glob.glob(new_name))
for video in video_lists:
    video = video.replace('\\', '/')
    video_name = video.split('/')[-1]
    video_name = video_name.split('.')[0]
    img_folder = os.path.join(args.img_path,video_name)
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    command = 'ffmpeg -i {0} -vsync 0 {1}/%3d.png -y'.format(video, img_folder)
    os.system(command)

#python3 video2img.py --mp4-path /home/data/videos/LDR_540p --img-path ./train_lr
#python3 video2img.py --mp4-path /home/data/videos/SDR_4K --img-path ./train_hr
#python3 video2img.py --mp4-path /home/data/LDR_540p --img-path ./test_lr