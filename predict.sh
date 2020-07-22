#!/usr/bin/env bash


docker run --gpus all --shm-size=16g --rm -it -u $(id -u):$(id -g)  \
-v $(pwd):/work -v $(pwd)/.cache:/.cache -v /work/data:/data \
pytorch:latest \
python /work/predict.py \
--result_path /work/result/efb5 \
--submission_file submission_efb5_tta.csv \
--TTA
