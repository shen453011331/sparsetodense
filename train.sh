#!/bin/bash


batch_size=2
workers=4
log_dir="./logs"
model='sparse_to_dense'
base_path=/home/titan/results/mode=sparse+photo.w1=0.1.w2=0.1.input=rgbd.resnet34.criterion=l2.lr=1e-05.bs=2.wd=0.pretrained=False.jitter=0.1.time=2020-
time_date=02-26@21-44/
checkpoint_file=checkpoint-8-3000.pth.tar
checkpoint_path=$base_path$time_date$checkpoint_file
#train!
python main.py --train-mode sparse+photo \
               --input rgbd  \
               -w $workers \
               -b $batch_size  \
               --savemodel $log_dir \
               --model $model \
               --resume $checkpoint_path \

