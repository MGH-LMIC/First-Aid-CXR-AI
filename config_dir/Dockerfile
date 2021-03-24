
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    build-essential \
    libgtk2.0-dev \
    unzip \
    vim \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
                            
ENV SHELL=/bin/bash

COPY ./ /usr/app/
WORKDIR /usr/app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
