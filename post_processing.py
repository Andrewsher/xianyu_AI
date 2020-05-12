import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

import os

def post_processing(img, kernel_open=24, kernel_close=36):
    '''
    图像后处理
    :param img: np.array, [h, w, 4], np.uint8
    :param kernel_open:
    :param kernel_close:
    :return:
    '''
    # 闭运算填充孔洞
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((kernel_close, kernel_close), dtype=np.uint8))
    # 开运算去除小的连通集
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((kernel_open, kernel_open), dtype=np.uint8))
    return img
    # new_img = []
    # for i in range(4):
    #     # 闭运算填充孔洞
    #     tmp_img = cv2.morphologyEx(img[:, :, i], cv2.MORPH_CLOSE, np.ones((kernel_close, kernel_close), dtype=np.uint8))
    #     # 开运算去除小的连通集
    #     tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_OPEN, np.ones((kernel_open, kernel_open), dtype=np.uint8))
    #     new_img.append(tmp_img)
    # return np.array(new_img).transpose([1, 2, 0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-p', '--path', default='output-2-predict', type=str)
    args = parser.parse_args()
    os.makedirs('processed_predict', exist_ok=True)
    for image_name in tqdm(os.listdir(path=args.path)):
        image = Image.open(os.path.join('predict', image_name))
        print(image.size)
        image_array = np.array(image)
        # image = to_categorical(image, num_classes=4)
        print(image_array.shape)
        # for i in range(4):
        tmp_image = post_processing(np.uint8(image_array*64), kernel_close=75, kernel_open=75)
        print(tmp_image.shape)
        # tmp_image = np.uint8(tmp_image // 55)
        Image.fromarray(np.uint8(tmp_image)).resize((tmp_image.shape[1]//10, tmp_image.shape[0]//10), resample=Image.ANTIALIAS).save('resized_predict_image/' + image_name)
        Image.fromarray(np.uint8(tmp_image//60)).save('processed_predict/' + image_name)
        # tmp_image = cv2.resize(np.uint8(tmp_image*64), (tmp_image.shape[0]//10, tmp_image.shape[1]//10))
        # cv2.imwrite('resized_predict_image/' + image_name, tmp_image)
