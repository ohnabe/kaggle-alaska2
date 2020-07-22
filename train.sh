#!/usr/bin/env bash

#DISPLAY=":0"

docker run --gpus all --shm-size=16g --rm -it -e DISPLAY=$DISPLAY -u $(id -u):$(id -g)  \
-v $(pwd):/work -v $(pwd)/.cache:/.cache \
pytorch:latest \
python /work/train.py


#-v /media/realnabe/ec33b836-a439-4ee0-99b8-78c38cfb7d1b/Kaggle/ALASKA2/data:/data \