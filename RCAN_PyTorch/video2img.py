import glob,os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mp4-path', type=str, required=True)
parser.add_argument('--img-path', type=str, required=True)
args = parser.parse_args()


if not os.path.exists(args.img_path):
    os.mkdir(args.img_path)
new_name = args.mp4_path + '/*.mp4'
video_lists = sorted(glob.glob(new_name))
for video in video_lists:
    video_name = video.split('/')[-1]
    video_name = video_name.split('.')[0]
    img_folder = os.path.join(args.img_path,video_name)
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    command = 'ffmpeg -i {0} -vsync 0 {1}/%3d.png -y'.format(video, img_folder)
    os.system(command)

#python3 video2img.py --mp4-path ./SDR_540p --img-path ./data/test_lr
#python3 video2img.py --mp4-path ./videos/gt --img-path ./data/hr
#python3 video2img.py --mp4-path ./videos/X4 --img-path ./data/lr