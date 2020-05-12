# README

2019 年县域农业大脑AI挑战赛（初赛）解决方案

requirements:

* keras
* cv2
* PIL
* segmentation_models
* imgaug

步骤：

1. 使用一种crop方法进行crop，例如使用resized_crop

``` bash
python crop/resized_crop.py --data-file-path /data/data/xianyu_AI/jingwei_round1_train_20190619 --target-dir /data/data/xianyu_AI/resized_crop --crop-size 1024 --crop-num 10000
```
如果使用了不做resize的crop方法，需要去掉模型中最后的两个上采样层。

2. 训练模型
``` python
python train.py --data-file-path /data/data/xianyu_AI/resized_crop --gpu 0 --log-dir output-1 --backbone resnet18 --batch-size 8 --lr 0.0001 --allowed-image-size 1024 --pretrained-weights-file imagenet
```

3. fine-tune（可选）
``` python
python train_stage_3.py --data-file-path /data/data/xianyu_AI/resized_crop --gpu 0 --log-dir output-2 --backbone resnet18 --batch-size 8 --lr 0.0001 --epochs 40 --lr 0.0001 --allowed-image-size 1024 --pretrained-weights-file output-1/trained_final_weights.h5
```

4. 预测
``` python
python predict.py --data-file-path /data/data/xianyu_AI/jingwei_round1_test_a_20190619 --target-path output-2-predict --backbone resnet18 --gpu 0 --allowed-image-size 1024 --pretrained-weights-file output-2/trained_final_weights.h5
python visual_predicts.py --root-path output-2-predict --path predict
```

5. 后处理（可选）
``` python
python post_processing.py --path output-2-predict/predict
```
