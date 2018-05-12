FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y python3 python3-pip

ADD requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

ADD snet /app
