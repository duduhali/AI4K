import os
import cv2
import argparse

def cutCenter(png_path, cut_path, scale):
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)
        print('create floder', cut_path)

    for folder in os.listdir(png_path):
        png_folder = os.path.join(png_path, folder)
        cut_folder = os.path.join(cut_path, folder)
        if not os.path.exists(cut_folder):
            os.makedirs(cut_folder)
            print('create floder',cut_folder)

        for file in os.listdir(png_folder):
            src_file = os.path.join(png_folder, file)  # J:/AI+4K/pngs\gt\100913730001.png
            dst_file = os.path.join(cut_folder, file)  # J:/AI+4K/pngs_cut20\gt\100913730001.png
            # (960, 540)
            # (3840, 2160)
            img = cv2.imread(src_file, 1)
            height, width, _ = img.shape
            # print(height,width)
            tag_h, tag_w = int(height / (scale*2)), int(width / (scale*2))
            # scale*2份，取中间的两份

            dst = img[tag_h * (scale-1):tag_h * (scale+1), tag_w * (scale-1):tag_w * (scale+1)]
            # print(dst.shape)
            cv2.imwrite(dst_file, dst, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # png 是无损压缩  范围0-9，为9 时压缩比最高

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-path', type=str, required=True)
    parser.add_argument('--cut-path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=10)
    args = parser.parse_args()

    cutCenter(args.png_path, args.cut_path, args.scale)

    #python3 prepare/cut_img.py --png-path E:/test/pngs --cut-path E:/test/cut_pngs --scale 10
