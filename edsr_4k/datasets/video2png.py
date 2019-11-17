import os
import subprocess



png_path="E:/AI+4K/pngs"
video_path = "E:/AI+4K/videos"


#建立图片的目录
for folder in os.listdir(video_path):
    for x_path in [png_path]:
        the_folder = os.path.join(x_path,folder)
        if os.path.exists(the_folder) == False:
            os.makedirs(the_folder)
            print(the_folder)

    #视频转图片
    for file in os.listdir('{0}/{1}'.format(video_path,folder)):
        folder_file = '{0}/{1}/{2}'.format(video_path,folder,file) #videos/gt/10091373.mp4
        file_name,extend = os.path.splitext(file)   #  '10091373'  '.mp4'
        # log_name = '{0}/{1}/{2}.log'.format(log_path,folder,file_name) #logs/gt/10091373.log

        cmd_png = 'ffmpeg -i {0} -vsync 0 {1}/{2}/{3}%4d.png -y'.format(folder_file, png_path,folder,file_name)
        print(cmd_png) # ffmpeg -i ./videos/gt/10091373.mp4 -vsync 0 ./pngs/gt/10091373%4d.png -y
                        # ffmpeg -i E:/AI+4K/videos/gt/10099858.mp4 -vsync 0 E:/AI+4K/pngs/gt/10099858%4d.png -y
        process_png = subprocess.Popen(cmd_png, shell=True)
        process_png.wait()





