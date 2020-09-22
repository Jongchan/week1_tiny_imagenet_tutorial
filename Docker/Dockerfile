FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN pip install --upgrade pip
RUN pip install ipdb wandb opencv-python==4.1.2.30

RUN apt-get update
RUN apt-get install -y wget curl unzip sudo vim libgtk2.0-dev