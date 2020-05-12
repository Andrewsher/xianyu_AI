from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

import os

def order_crop(args):
    # mkdir
    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    if not os.path.isdir(os.path.join(args.target_dir, 'image')):
        os.mkdir(os.path.join(args.target_dir, 'image'))
    if not os.path.isdir(os.path.join(args.target_dir, 'image', str(args.crop_size))):
        os.mkdir(os.path.join(args.target_dir, 'image', str(args.crop_size)))
    if not os.path.isdir(os.path.join(args.target_dir, 'image', str(args.crop_size), 'origin')):
        os.mkdir(os.path.join(args.target_dir, 'image', str(args.crop_size), 'origin'))
    if not os.path.isdir(os.path.join(args.target_dir, 'image', str(args.crop_size), 'rotate')):
        os.mkdir(os.path.join(args.target_dir, 'image', str(args.crop_size), 'rotate'))

    if not os.path.isdir(os.path.join(args.target_dir, 'annotation')):
        os.mkdir(os.path.join(args.target_dir, 'annotation'))
    if not os.path.isdir(os.path.join(args.target_dir, 'annotation', str(args.crop_size))):
        os.mkdir(os.path.join(args.target_dir, 'annotation', str(args.crop_size)))
    if not os.path.isdir(os.path.join(args.target_dir, 'annotation', str(args.crop_size), 'origin')):
        os.mkdir(os.path.join(args.target_dir, 'annotation', str(args.crop_size), 'origin'))
    if not os.path.isdir(os.path.join(args.target_dir, 'annotation', str(args.crop_size), 'rotate')):
        os.mkdir(os.path.join(args.target_dir, 'annotation', str(args.crop_size), 'rotate'))

    # 初始化
    image_names = ['image_1', 'image_2']
    Image.MAX_IMAGE_PIXELS = 100000000000000
    opened_images = []
    opened_labels = []
    hs = [0, 0]
    ws = [0, 0]
    allowed_size = args.crop_size
    step = allowed_size//2

    # 打开图片
    for image_name in image_names:
        opened_images.append(Image.open(os.path.join(args.data_file_path, image_name + '.png')))
        opened_labels.append(Image.open(os.path.join(args.data_file_path, image_name + '_label.png')))
    for i in range(len(opened_images)):
        hs[i] = opened_images[i].size[0]
        ws[i] = opened_images[i].size[1]

    for i in range(2):
        image_idx = i
        cy = 0
        while cy + allowed_size < hs[i]:
            cx = 0
            while cx + allowed_size < ws[i]:
                current_img = opened_images[image_idx].crop((cx, cy, cx+allowed_size, cy+allowed_size))
                if current_img.getpixel((allowed_size//2, allowed_size//2))[3] != 0:
                    img_name = '{}_{}_{}'.format(i, cx, cy)
                    current_img.save(os.path.join(
                        args.target_dir, 'image', str(allowed_size), 'origin', img_name + '.png'))
                    current_label = opened_labels[image_idx].crop((cx, cy, cx+allowed_size, cy+allowed_size))
                    current_label.point(lambda i : i * 64).save(os.path.join(
                        args.target_dir, 'annotation', str(allowed_size), 'origin', img_name + '_label.png'))
                    cx += step
                else:
                    cx += step
            cy +=step


def parse_args():
    parser = argparse.ArgumentParser(description='random_crop')
    # config
    parser.add_argument('-d', '--data-file-path', default='/home/amax/yzg/baseline-unet/jingwei_round1_train_20190619', type=str)
    parser.add_argument('-l', '--target-dir', default='/home/amax/yzg/baseline-unet/order_crop', type=str)
    parser.add_argument('-s', '--crop-size', default=2048, type=int)
    parser.add_argument('-n', '--crop-num', default=50000, type=int)
    parser.add_argument('-r', '--rotate', default=True, type=bool)
    parser.add_argument('-b', '--balance', default=False, type=bool)

    args = parser.parse_args()

    print('data file path = ', args.data_file_path, '\ntarget dir = ', args.target_dir,
          '\ncrop size = ', args.crop_size, '\ncrop num = ', args.crop_num,
          '\nallow rotate = ', args.rotate, '\nbalance = ', args.balance)

    return args


if __name__ == '__main__':
    args = parse_args()
    order_crop(args)
