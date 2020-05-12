import keras
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
from keras.losses import categorical_crossentropy
import os
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import iou_score, jaccard_score, f1_score, f2_score, dice_score
from segmentation_models.utils import set_trainable
import argparse
from keras.metrics import categorical_accuracy
from imageio import imread
from PIL import Image
# import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = 100000000000000000
import math
import cv2
from tqdm import tqdm


from loss import dice_0, dice_1, dice_2, dice_3, iou_0, iou_1, iou_2, iou_3, iou_loss_3_sum, iou_loss_weight_sum
from models import create_unet, create_unet_nonlocal, create_exist_model, create_backbone_unet, create_class_unet
from datas import DataGenerator, split_train_val
from post_processing import post_processing


def get_patch(img, args):
    patch_size = args.allowed_image_size
    center_size = args.allowed_image_size // 2
    l_margin = math.floor((patch_size - center_size) / 2)
    r_margin = math.ceil((patch_size - center_size) / 2)
    step = center_size

    row = img.shape[0]
    col = img.shape[1]
    yield math.ceil(row / step), math.ceil(col / step)

    top = l_margin
    bottom = math.ceil((row) / step) * step - (row) + l_margin
    left = l_margin
    right = math.ceil((col) / step) * step - (col) + l_margin
    print('padding...')
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
    # patch_all = np.zeros(shape=(math.ceil(row / step) * math.ceil(col / step), patch_size, patch_size, 4), dtype=np.float16)
    y = []
    count = 1
    for i in range(math.ceil(row / step)):
        for j in range(math.ceil(col / step)):
            if len(y) >= 32:
                yield np.array(y)
                y = []
                # count = 1
            # else:
            #     count += 1
            y_tmp = np.array(img[i * step:i * step + patch_size, j * step:j * step + patch_size, :])
            y_tmp = Image.fromarray(np.uint8(y_tmp)).resize((patch_size//4, patch_size//4), resample=Image.BILINEAR)
            y.append(np.array(y_tmp))
            # patch_all[i * math.ceil(col / step) + j, :, :, :] = img[i * step:i * step + patch_size, j * step:j * step + patch_size, :] / 255.
            # yield np.array([img[i * step:i * step + patch_size, j * step:j * step + patch_size, 0:3] / 255.])

    # print('test')
    # patch_all = np.array(patch_all)
    if len(y) != 0:
        yield np.array(y)
    # return patch_all, math.ceil(row / step), math.ceil(col / step)



def patchToImg(label, img_shape, p_row, p_col, center_size):
    b, h, w, _ = label.shape
    img_y = np.zeros(img_shape, dtype=np.uint8)
    for i in tqdm(range(p_row)):
        for j in range(p_col):
            # if i != p_row - 1 and j != p_col - 1:
            #     current_label = label[i * p_col + j, :, :, :]
            #     current_label = np.argmax(current_label, axis=-1)
            #     current_label = current_label * (patch[i * p_col + j, :, :, 3] // 250)
            #     img_y[i * center_size:(i + 1) * center_size, j * center_size:(j + 1) * center_size] = current_label[(w-center_size)//2:(w-center_size)//2+center_size, (h-center_size)//2:(h-center_size)//2+center_size]
            # else:
            current_label = label[i * p_col + j, :, :, :]
            current_label = np.argmax(current_label, axis=-1)
            # current_label = current_label * (patch[i * p_col + j, :, :, 3] // 250)
            w1, h1 =img_y[i * center_size:(i + 1) * center_size, j * center_size:(j + 1) * center_size].shape
            img_y[i * center_size:(i + 1) * center_size, j * center_size:(j + 1) * center_size] = current_label[(w - center_size) // 2:(w - center_size) // 2 + w1,(h - center_size) // 2:(h - center_size) // 2 + h1]
    return img_y



def predict(args):

    if not os.path.isdir(args.target_path):
        os.mkdir(args.target_path)
    if not os.path.isdir(os.path.join(args.target_path, 'predict_each_epoch')):
        os.mkdir(os.path.join(args.target_path, 'predict_each_epoch'))


    K.clear_session()
    model = create_backbone_unet(input_shape=(args.allowed_image_size//4, args.allowed_image_size//4, 3), pretrained_weights_file=None, backbone=args.backbone)

    image_names = ['image_3', 'image_4']
    for image_name in image_names:
        image = imread(os.path.join(args.data_file_path, image_name + '.png'))

        for h5_file in os.listdir(args.pretrained_weights_file_dir):
            if h5_file.startswith('ep') and h5_file.endswith('.h5'):
                print(os.path.join(args.pretrained_weights_file_dir, h5_file))
                model.load_weights(os.path.join(args.pretrained_weights_file_dir, h5_file))

                f = get_patch(image, args)
                p_row, p_col = f.__next__()
                img_y = np.zeros(shape=image.shape[0:2], dtype=np.uint8)

                count = 0
                i = 0
                j = 0
                center_size = args.allowed_image_size // 2
                for patches in tqdm(f):
                    labels = model.predict(patches[:, :, :, 0:3] / 255.)
                    # print(len(np.where(labels[:, :, :, 1:3] >= 0.5)[0]), labels.shape)

                    # labels = np.array(labels * (patches[:, :, :, 3])[:, :, :, None], dtype=np.uint8)  # convert to [0, 255], uint8
                    labels = np.argmax(labels, axis=-1)

                    b, h, w = labels.shape
                    for label in labels:
                        # post_processing
                        # label = post_processing(label)
                        # write to image
                        # label = np.argmax(label, axis=-1)
                        w1, h1 = img_y[i * center_size:(i + 1) * center_size, j * center_size:(j + 1) * center_size].shape
                        img_y[i * center_size:(i + 1) * center_size, j * center_size:(j + 1) * center_size] = label[(w - center_size) // 2:(w - center_size) // 2 + w1,(h - center_size) // 2:(h - center_size) // 2 + h1]
                        j += 1
                        if j >= p_col:
                            j = 0
                            i += 1

                # labels = model.predict_generator(generator=f, steps=math.ceil(p_col * p_row / 32), verbose=1)
                # labels = np.argmax(labels, axis=-1)
                # labels = labels * (patches[:, :, :, 3] // 250)
                # predict = patchToImg(label=labels, img_shape=image.shape[0:2], p_row=p_row, p_col=p_col, center_size=args.allowed_image_size//2)
                h, w = img_y.shape
                Image.fromarray(np.uint8(img_y)).save(os.path.join(args.target_path, 'predict_each_epoch', image_name + '_predict_' + h5_file.split('-')[0] + '.png'))



def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-file-path', default='/data/data/xianyu_AI/jingwei_round1_test_a_20190619', type=str)
    parser.add_argument('-t', '--target-path',
                        default='.', type=str)
    parser.add_argument('-b', '--backbone', default='resnet18', type=str)
    parser.add_argument('-g', '--gpu', default='0', type=str, required=True)

    # Train Setting
    parser.add_argument('--allowed-image-size', default=1536, type=int)
    # Pretrain and Checkpoint
    parser.add_argument('--pretrained-weights-file-dir', default=None,
                        help='imagenet, file_dir or None')

    args = parser.parse_args()

    print('data file path = ', args.data_file_path, '\npretrained weights file dir= ', args.pretrained_weights_file_dir,
          '\nallowed image size = ', args.allowed_image_size,
          '\nbackbone = ', args.backbone)

    return args


if __name__ == '__main__':
    # get argers
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    predict(args)
