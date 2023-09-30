FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt install unzip

COPY . /latenciaga
WORKDIR /latenciaga

RUN python3 -m pip install .
