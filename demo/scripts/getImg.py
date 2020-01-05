import os
import subprocess



png_path="J:/img_540p"
video_path = "J:/SDR_540p"



#视频转图片
for file in os.listdir(video_path):
    folder_file = '{}/{}'.format(video_path,file) #videos/gt/10091373.mp4
    file_name,extend = os.path.splitext(file)   #  '10091373'  '.mp4'
    cmd_png = 'ffmpeg -i {} -vsync 0 {}/{}%4d.png -y'.format(folder_file, png_path,file_name)
    print(cmd_png) # ffmpeg -i ./videos/gt/10091373.mp4 -vsync 0 ./pngs/gt/10091373%4d.png -y
    process_png = subprocess.Popen(cmd_png, shell=True)
    process_png.wait()




