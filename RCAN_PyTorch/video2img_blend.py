#图片全都放在一个目录下
import os
import subprocess
import argparse

#视频转图片
def getImg(video_path,img_path,suffix):
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for file in os.listdir(video_path):
        folder_file = '{}/{}'.format(video_path,file) #videos/gt/10091373.mp4
        file_name,extend = os.path.splitext(file)   #  '10091373'  '.mp4'
        cmd_png = 'ffmpeg -i {} -vsync 0 {}/{}%{}d.png -y'.format(folder_file, img_path,file_name,suffix)
        print(cmd_png) # ffmpeg -i ./videos/gt/10091373.mp4 -vsync 0 ./pngs/gt/10091373%4d.png -y
        process_png = subprocess.Popen(cmd_png, shell=True)
        process_png.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video to picture')
    parser.add_argument('--video-path',"--vp", type=str, required=True)
    parser.add_argument('--img-path', "--ip",type=str, required=True)
    parser.add_argument('--suffix', type=int, default=4)
    args = parser.parse_args()

    getImg(args.video_path, args.img_path, args.suffix)

    #python video2img.py --video-path E:/test/videos/gt --img-path E:test/pngs/gt
    #python video2img.py --vp E:/test/videos/X4 --ip E:test/pngs/X4

    #python3 video2img_blend.py --vp /home/ubuntu/test/gt --ip ./img_hr
    #python3 video2img_blend.py --vp /home/ubuntu/test/X4 --ip ./img_lr
    #python3 video2img_blend.py --vp /home/ubuntu/SDR_540p --ip ./sdr_540p

