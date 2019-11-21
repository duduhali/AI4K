import os
import cv2

png_path="J:/AI+4K/pngs"
png_cut20 = "J:/AI+4K/pngs_cut20"

for folder in os.listdir(png_path):
    cut20_folder = os.path.join(png_cut20, folder)
    png_folder = os.path.join(png_path, folder)
    if os.path.exists(cut20_folder) == False:
        os.makedirs(cut20_folder)
        print(cut20_folder)

    for file in os.listdir(png_folder):
        src_file = os.path.join(png_folder,file)  # J:/AI+4K/pngs\gt\100913730001.png
        dst_file =  os.path.join(cut20_folder,file) #J:/AI+4K/pngs_cut20\gt\100913730001.png

        #(960, 540)
        # (3840, 2160)
        img = cv2.imread(src_file, 1)
        height,width,_ = img.shape
        # print(height,width)
        tag_h,tag_w = int(height/20),int(width/20)
        #20份，取中间的两份
        dst = img[tag_h*9:tag_h*11, tag_w*9:tag_w*11]
        print(dst.shape)
        cv2.imwrite(dst_file, dst, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # png 是无损压缩  范围0-9，为9 时压缩比最高

        # cv2.imshow("image", dst)
        # cv2.waitKey(0)

