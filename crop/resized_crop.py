''' unicode: utf-8'''

from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

import os


# 确定中心点
def get_center_location(img, h, w, size):
    center_x, center_y = np.random.randint(size//2, h - size//2), np.random.randint(size//2, w - size//2)
    while img.getpixel((center_x, center_y))[3] == 0:
        center_x, center_y = np.random.randint(size//2, h - size//2), np.random.randint(size//2, w - size//2)
    return center_x, center_y


def rotation_45_crop(img, center_x, center_y, w, h):
    '''
    对图像做倾斜裁切
    :param img:
    :param center_x:
    :param center_y:
    :param w: 宽
    :param h: 长
    :return:
    '''
    new_w, new_h = int(2 * w), int(2 * h)
    new_img = img.crop((center_x - new_w // 2, center_y - new_h // 2, center_x + new_w // 2, center_y + new_h // 2))
    rotated_img = new_img.rotate(45, resample=Image.BILINEAR, expand=True)
    rotated_w, rotated_h = rotated_img.size
    cropped_img = rotated_img.crop((rotated_w//2 - w//2, rotated_h//2 - h//2, rotated_w//2 + w//2, rotated_h//2 + h//2))
    return cropped_img


def random_crop(args):
    # 建立文件夹
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
    target_size = allowed_size // 4

    # 打开图片
    for image_name in image_names:
        opened_images.append(Image.open(os.path.join(args.data_file_path, image_name + '.png')))
        opened_labels.append(Image.open(os.path.join(args.data_file_path, image_name + '_label.png')))
    for i in range(len(opened_images)):
        hs[i] = opened_images[i].size[0]
        ws[i] = opened_images[i].size[1]

    # 随机裁切
    for i in tqdm(range(args.crop_num)):

        # 确定要切哪幅图
        if i < args.crop_num // 2:
            image_idx = 0
        else:
            image_idx = 1
        # 确定中心点
        center_x, center_y = get_center_location(img=opened_images[image_idx], h=hs[image_idx], w=ws[image_idx],
                                                 size=allowed_size)
        if args.balance == True:
            while i % 4 != opened_labels[image_idx].getpixel((center_x, center_y)):
                center_x, center_y = get_center_location(img=opened_images[image_idx], h=hs[image_idx], w=ws[image_idx],
                                                         size=allowed_size)

        # 水平裁切或旋转裁切
        if args.rotate == True:
            if np.random.randint(0, 2) == 0:
                current_img = opened_images[image_idx].crop(
                    (center_x - allowed_size // 2, center_y - allowed_size // 2,
                     center_x + allowed_size // 2, center_y + allowed_size // 2))
                current_img.resize((target_size, target_size), resample=Image.BILINEAR).save(os.path.join(args.target_dir, 'image', str(args.crop_size), 'origin', str(i) + '.png'))
                current_label = opened_labels[image_idx].crop(
                    (center_x - allowed_size // 2, center_y - allowed_size // 2,
                     center_x + allowed_size // 2, center_y + allowed_size // 2))
                current_label.point(lambda i: i * 64).save(os.path.join(args.target_dir, 'annotation', str(args.crop_size), 'origin', str(i) + '_label.png'))
            else:
                current_img = rotation_45_crop(opened_images[image_idx], center_x, center_y, allowed_size,
                                               allowed_size)
                current_img.resize((target_size, target_size), resample=Image.BILINEAR).save(os.path.join(args.target_dir, 'image', str(args.crop_size), 'rotate', str(i) + '.png'))
                current_label = rotation_45_crop(opened_labels[image_idx], center_x, center_y, allowed_size,
                                                 allowed_size)
                current_label.point(lambda i: i * 64).save(os.path.join(args.target_dir, 'annotation', str(args.crop_size), 'rotate', str(i) + '_label.png'))
        else:
            current_img = opened_images[image_idx].crop((center_x - allowed_size // 2, center_y - allowed_size // 2,
                                                         center_x + allowed_size // 2,
                                                         center_y + allowed_size // 2))
            current_img.resize((target_size, target_size), resample=Image.BILINEAR).save(os.path.join(args.target_dir, 'image', str(args.crop_size), 'origin', str(i) + '.png'))
            current_label = opened_labels[image_idx].crop(
                (center_x - allowed_size // 2, center_y - allowed_size // 2,
                 center_x + allowed_size // 2, center_y + allowed_size // 2))
            current_label.point(lambda i: i * 64).save(
                os.path.join(args.target_dir, 'annotation', str(args.crop_size), 'origin', str(i) + '_label.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='random_crop')
    # config
    parser.add_argument('-d', '--data-file-path', default='/data/data/xianyu_AI/jingwei_round1_train_20190619', type=str)
    parser.add_argument('-l', '--target-dir', default='/data/data/xianyu_AI/resized_crop', type=str)
    parser.add_argument('-s', '--crop-size', default=1536, type=int)
    parser.add_argument('-n', '--crop-num', default=10000, type=int)
    parser.add_argument('-r', '--rotate', default=True, type=bool)
    parser.add_argument('-b', '--balance', default=False, type=bool)

    args = parser.parse_args()

    print('data file path = ', args.data_file_path, '\ntarget dir = ', args.target_dir,
          '\ncrop size = ', args.crop_size, '\ncrop num = ', args.crop_num,
          '\nallow rotate = ', args.rotate, '\nbalance = ', args.balance)

    return args


if __name__ == '__main__':
    args = parse_args()
    random_crop(args)

