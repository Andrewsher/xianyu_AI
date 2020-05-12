import keras
import keras.backend as K
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.utils import multi_gpu_model
import os
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import iou_score, jaccard_score, f1_score, f2_score, dice_score
from segmentation_models.utils import set_trainable
import argparse

from loss import dice_0, dice_1, dice_2, dice_3, iou_0, iou_1, iou_2, iou_3, iou_loss_3_sum, iou_loss_weight_sum, \
    focal_loss, iou_3_sum, dice_loss, dice_iou_loss
from lovasz_losses_tf import lovasz_softmax
from models import create_unet, create_unet_nonlocal, create_exist_model, create_backbone_unet, \
    create_multi_branch_unet, create_class_unet
from datas import DataGenerator, split_train_val, MultiBranch_DataGenerator, DataAugGenerator, DateWithClassGenerator


def train(train_set, val_set, args):

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    # Get callbacks
    checkpoint = ModelCheckpoint(args.log_dir + '/ep={epoch:03d}-loss={loss:.3f}-val_loss={val_loss:.3f}.h5', verbose=1,
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_output_seg_iou_3_sum', factor=0.2, min_delta=1e-2, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_output_seg_iou_3_sum', min_delta=1e-2, patience=10, verbose=1)
    csv_logger = CSVLogger(args.log_dir + '/record.csv')
    tensorboard = TensorBoard(log_dir=args.log_dir)

    # Train the proposed model
    K.clear_session()
    model = create_backbone_unet(input_shape=(args.allowed_image_size//4, args.allowed_image_size//4, 3), pretrained_weights_file=args.pretrained_weights_file, backbone=args.backbone)
    model.compile(optimizer=Adam(lr=args.lr), loss=[jaccard_loss, binary_crossentropy],
                  metrics=[iou_score, dice_score, dice_0, dice_1, dice_2, dice_3, iou_0, iou_1, iou_2, iou_3, iou_3_sum])
    # pretrain
    # model.fit_generator(
    #     generator=DataGenerator(dir_path=args.data_file_path, batch_size=args.batch_size * 2, img_set=train_set,
    #                             shuffle=args.shuffle),
    #     validation_data=DataGenerator(dir_path=args.data_file_path, batch_size=args.batch_size, img_set=val_set,
    #                                   shuffle=True),
    #     epochs=4,
    #     initial_epoch=0,
    #     callbacks=[checkpoint, tensorboard, reduce_lr, reduce_lr, early_stopping, csv_logger],
    #     workers=4,
    #     use_multiprocessing=True)
    # train
    set_trainable(model=model)
    model.compile(optimizer=Adam(lr=args.lr), loss=[jaccard_loss, binary_crossentropy],
                  metrics=[iou_score, dice_score, dice_0, dice_1, dice_2, dice_3, iou_0, iou_1, iou_2, iou_3, iou_3_sum])
    model.fit_generator(
        generator=DataGenerator(dir_path=args.data_file_path, batch_size=args.batch_size, img_set=train_set, shuffle=args.shuffle),
        validation_data=DataGenerator(dir_path=args.data_file_path, batch_size=args.batch_size, img_set=val_set, shuffle=True),
        epochs=args.epochs,
        initial_epoch=0,
        callbacks=[checkpoint, tensorboard, reduce_lr, early_stopping, csv_logger],
        workers=8,
        use_multiprocessing=False)
    model.save_weights(os.path.join(args.log_dir,  'trained_final_weights.h5'))

    # exit training
    K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-file-path', default='/home/qkh/hdd1/data/xianyu_AI/unbalance_crop', type=str)
    parser.add_argument('-g', '--gpu', default='7', type=str)
    parser.add_argument('-l', '--log-dir', default='output-0/', type=str)
    parser.add_argument('-b', '--backbone', default='resnet18', type=str)
    # Train Setting
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--allowed-image-size', default=1024, type=int)
    # Pretrain and Checkpoint
    parser.add_argument('--pretrained-weights-file', default=None,
                        help='imagenet, file_dir or None')
    parser.add_argument('--val-rate', default=0.2, type=float)
    parser.add_argument('--shuffle', default=True, type=bool)

    args = parser.parse_args()

    print('data file path = ', args.data_file_path, '\npretrained weights file = ', args.pretrained_weights_file,
          '\nallowed image size = ', args.allowed_image_size, '\nbackbone = ', args.backbone,
          '\nbatch size = ', args.batch_size, '\nlog dir = ', args.log_dir,
          '\nmax epochs = ', args.epochs, '\nlr = ', args.lr,
          '\nvalidation rate = ', args.val_rate), '\nshuffle = ', args.shuffle

    return args


if __name__ == '__main__':
    # get argers
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_set, val_set = split_train_val(dir_path=args.data_file_path, size=args.allowed_image_size,
                                         val_rate=args.val_rate, shuffle=True)
    train(args=args, train_set=train_set, val_set=val_set)


