'''
读取数据
'''

'''
--- image
 |--- 384
   |--- origin
   |--- rotate
     |--- xxx.png
     |--- xxx.png
 |--- 256
 |--- 160
--- annotation
 |--- 384
   |--- origin
   |--- rotate
     |--- xxx_label.png
     |--- xxx_label.png
'''

import numpy as np
from skimage import io
import keras
from keras.utils import to_categorical
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage

import os


def split_train_val(dir_path, size=384, val_rate=0.25, shuffle=False):
    img_url = []
    # 载入文件相对路径，不包含.png后缀
    for img in os.listdir(os.path.join(dir_path, 'annotation', str(size), 'origin')):
        img_url.append(os.path.join(str(size), 'origin', img.split('_')[0]))
    if os.path.isdir(os.path.join(dir_path, 'annotation', str(size), 'rotate')):
        for img in os.listdir(os.path.join(dir_path, 'annotation', str(size), 'rotate')):
            img_url.append(os.path.join(str(size), 'rotate', img.split('_')[0]))
    # 打乱次序
    if shuffle == True:
        np.random.shuffle(img_url)
    total_num = len(img_url)
    val_num = int(total_num * val_rate)
    # 划分验证集
    val_set = img_url[0: val_num//2] + img_url[total_num - val_num//2 : total_num]
    train_set = img_url

    return train_set, val_set


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dir_path, batch_size, img_set, shuffle=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.img_set = img_set
        self.shuffle = shuffle
        if self.shuffle == True:
            np.random.shuffle(self.img_set)

    def __len__(self):
        return int(np.ceil(len(self.img_set) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_urls = self.img_set[index * self.batch_size : (index+1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for url in batch_urls:
            current_image = np.array(Image.open(os.path.join(self.dir_path, 'image', url + '.png')))
            # current_image = self.seq.augment_image(current_image)
            batch_images.append(current_image)
            current_label = np.array(Image.open(os.path.join(self.dir_path, 'annotation', url + '_label.png'))) // 60
            batch_labels.append(current_label)
        batch_images = np.array(batch_images)[:, :, :, :3] / 255.
        batch_labels = np.array(to_categorical(batch_labels, num_classes=4))

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.img_set)




class MultiBranch_DataGenerator(keras.utils.Sequence):
    def __init__(self, dir_path, batch_size, img_set, shuffle=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.img_set = img_set
        self.shuffle = shuffle
        if self.shuffle == True:
            np.random.shuffle(self.img_set)

    def __len__(self):
        return int(np.ceil(len(self.img_set) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_urls = self.img_set[index * self.batch_size : (index+1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for url in batch_urls:
            current_image = np.array(Image.open(os.path.join(self.dir_path, 'image', url + '.png')))
            batch_images.append(current_image)
            current_label = np.array(Image.open(os.path.join(self.dir_path, 'annotation', url + '_label.png'))) // 60
            batch_labels.append(current_label)
        batch_images = np.array(batch_images)[:, :, :, :3] / 255.
        batch_labels_1 = np.array(to_categorical(batch_labels, num_classes=4))
        batch_labels_2 = np.array(to_categorical(np.array(batch_labels) > 0.5, num_classes=2))

        return batch_images, [batch_labels_1, batch_labels_2]

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.img_set)



class DataAugGenerator(keras.utils.Sequence):
    def __init__(self, dir_path, batch_size, img_set, shuffle=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.img_set = img_set
        self.shuffle = shuffle
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90(k=[0, 1, 2, 3]),
            # iaa.CropAndPad(percent=(-0.2, 0.2))
            iaa.AllChannelsHistogramEqualization()
        ])
        if self.shuffle == True:
            np.random.shuffle(self.img_set)

    def __len__(self):
        return int(np.ceil(len(self.img_set) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_urls = self.img_set[index * self.batch_size : (index+1) * self.batch_size]
        batch_images = []
        batch_labels = []
        det = self.seq.to_deterministic()
        for url in batch_urls:
            current_image = np.array(Image.open(os.path.join(self.dir_path, 'image', url + '.png')))
            current_image = det.augment_image(current_image)
            batch_images.append(current_image)

            current_label = np.array(Image.open(os.path.join(self.dir_path, 'annotation', url + '_label.png'))) // 60
            current_label = SegmentationMapOnImage(current_label, shape=current_label.shape, nb_classes=4)
            current_label = det.augment_segmentation_maps(current_label)
            current_label = current_label.get_arr_int()
            batch_labels.append(current_label)
        batch_images = np.array(batch_images)[:, :, :, :3] / 255.
        batch_labels = np.array(to_categorical(batch_labels, num_classes=4))

        # det = self.seq.to_deterministic()
        # batch_images = det.augment_image(batch_images)
        # batch_labels = det.augment_segmentation_maps(batch_labels)
        # batch_images, batch_labels = self.seq(image=batch_images, segmentation_maps=batch_labels)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.img_set)


class DateWithClassGenerator(keras.utils.Sequence):
    def __init__(self, dir_path, batch_size, img_set, shuffle=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.img_set = img_set
        self.shuffle = shuffle
        if self.shuffle == True:
            np.random.shuffle(self.img_set)

    def __len__(self):
        return int(np.ceil(len(self.img_set) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_urls = self.img_set[index * self.batch_size : (index+1) * self.batch_size]
        batch_images = []
        batch_labels = []
        batch_classes = []
        for url in batch_urls:
            current_image = np.array(Image.open(os.path.join(self.dir_path, 'image', url + '.png')))
            # current_image = self.seq.augment_image(current_image)
            batch_images.append(current_image)
            current_label = np.array(Image.open(os.path.join(self.dir_path, 'annotation', url + '_label.png'))) // 60
            batch_labels.append(current_label)
            current_classes = np.array([len(np.where(current_label == 0)[1]), len(np.where(current_label == 1)[1]),
                                 len(np.where(current_label == 2)[1]), len(np.where(current_label == 3)[1])]) > 0 # 有 or 没有
            batch_classes.append(current_classes)
        batch_images = np.array(batch_images)[:, :, :, :3] / 255.
        batch_labels = np.array(to_categorical(batch_labels, num_classes=4))
        batch_classes = np.array(batch_classes, dtype=np.float)

        return batch_images, [batch_labels, batch_classes]

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.img_set)


if __name__ == "__main__":
    pass




