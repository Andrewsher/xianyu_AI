#!/usr/bin/env bash
# crop
python crop/resized_crop.py --data-file-path /data/data/xianyu_AI/jingwei_round1_train_20190619 --target-dir /data/data/xianyu_AI/resized_crop --crop-size 1024 --crop-num 10000

# train
python train.py --data-file-path /data/data/xianyu_AI/resized_crop --gpu 0 --log-dir resnet18-output-1 --backbone resnet18 --batch-size 8 --lr 0.0001 --allowed-image-size 1024 --pretrained-weights-file imagenet
# fine-tune with snapshot
# 取第1个epoch的结果。最后一个epoch的效果很差
python train_stage_3.py --data-file-path /data/data/xianyu_AI/resized_crop --gpu 0 --log-dir resnet18-output-2 --backbone resnet18 --batch-size 8 --lr 0.0001 --epochs 40 --lr 0.0001 --allowed-image-size 1024 --pretrained-weights-file resnet18-output-1/ep\=001-loss\=0.792-val_loss\=0.736.h5
