#FROM pytorch/pytorch:latest
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY 0

RUN pip install --upgrade pip &&\
    pip install efficientnet_pytorch pytorch-pfn-extras albumentations pandas scikit-learn gdown &&\
    apt-get update -y &&\
    apt-get install -y python-opencv git &&\
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* &&\
    git clone https://github.com/NVIDIA/apex

WORKDIR apex
RUN apt-get update -y &&\
    apt-get install -y tzdata &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
