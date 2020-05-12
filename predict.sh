#!/usr/bin/env bash
# predict
python predict.py --data-file-path /data/data/xianyu_AI/jingwei_round1_test_a_20190619 --target-path resnet18-output-2-predict --backbone resnet18 --gpu 0 --allowed-image-size 1024 --pretrained-weights-file resnet18-output-2/trained_final_weights.h5
# visualize
python visial_predicts.py --root-path resnet18-output-2-predict --path predict