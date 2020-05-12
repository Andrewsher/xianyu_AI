from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import os

parser = argparse.ArgumentParser(description='train')
# config
parser.add_argument('-r', '--root-path', default='resnet18-output-41-predict', type=str)
parser.add_argument('-p', '--path', default='predict_each_epoch', type=str)
args = parser.parse_args()

root_path = args.root_path
path = args.path

if not os.path.isdir(os.path.join(root_path, 'resized_predict_image')):
    os.mkdir(os.path.join(root_path, 'resized_predict_image'))

Image.MAX_IMAGE_PIXELS = 100000000000000

# images = 'predict_each_epoch'

for image in tqdm(os.listdir(os.path.join(root_path, path))):
    # a = Image.open(os.path.join(path, image + '.png'))
    # w, h = a.size
    # a.resize((w//10, h//10), Image.ANTIALIAS).save(os.path.join('resized_predict_image', image + '.png'))

    b = Image.open(os.path.join(root_path, path, image))
    b = b.point(lambda i: i * 64)
    w, h = b.size
    b.resize((w//10, h//10), Image.ANTIALIAS).save(os.path.join(root_path, 'resized_predict_image', image))

