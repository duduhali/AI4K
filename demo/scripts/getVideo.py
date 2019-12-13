#把图片转化为符合提交要求的视频的工具

import os, subprocess
import cv2

#把一堆图片转换为多个视频
def transform(img_path,video_path):
    name_list = set()
    sum_dict = dict()
    for img_file in os.listdir(img_path):
        one_name  = img_file[0:8]
        tmp_sum = sum_dict.get(one_name,0)
        sum_dict[one_name] = tmp_sum+1
        name_list.add(one_name)
    print(len(name_list))
    for k,v in sum_dict.items():
        if v != 100:
            raise Exception('%s 不等于100帧'%k)

    for name in name_list:
        src = '{}/{}%4d.png'.format(img_path,name)
        dst = '{}/{}.mp4'.format(video_path,name)
        if os.path.exists(dst):
            continue
        cmd_encoder = 'ffmpeg -r 24000/1001 -i '+ src + '  -vcodec libx265 -pix_fmt yuv422p -crf 10 ' + dst
        print(cmd_encoder)
        # ffmpeg -r 24000/1001 -i J:/output/16536366%4d.png  -vcodec libx265 -pix_fmt yuv422p -crf 10 J:/submission/16536366.mp4
        process_encoder = subprocess.Popen(cmd_encoder, shell=True)
        process_encoder.wait()

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



if __name__ == '__main__':
    img_path = 'J:/img_H'
    video_path = 'J:/ESPCN'
    # transform(img_path, video_path) #合成视频
    getVideoSize(video_path) #检测视频



