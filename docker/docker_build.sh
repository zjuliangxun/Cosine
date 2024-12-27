#!/usr/bin/env bash

docker build --network host -t isaacgym:v0 -f docker/Dockerfile .
# docker build --network host -t isaacgym:v1 -f docker/Dockerfile-v1 .


# https://pytorch.org/get-started/previous-versions/
# https://hub.docker.com/layers/pytorch/pytorch/2.0.1-cuda11.7-cudnn8-devel/images/sha256-4f66166dd757752a6a6a9284686b4078e92337cd9d12d2e14d2d46274dfa9048?context=explore
# https://data.code.gouv.fr/usage/docker/nvidia%2Fcuda
export DOCKER_BUILDKIT=1
docker build \
    --network host \
    -t isaacgym:4090-v0 \
    --build-arg CUDA_DOCKER_VERSION=11.7.1-devel-ubuntu20.04 \
    --build-arg PYTORCH_VERSION=1.13.1+cu117 \
    --build-arg TORCHVISION_VERSION=0.14.1+cu117 \
    --build-arg NCCL_VERSION=2.13.4-1+cuda11.7 \
    --build-arg TENSORFLOW_VERSION=2.9.2 \
    --build-arg CUDNN_VERSION=8.5.0.96-1+cuda11.7 \
    --build-arg PYSPARK_PACKAGE=pyspark==3.3.0 \
    --build-arg SPARK_PACKAGE=spark-3.3.0/spark-3.3.0-bin-hadoop2.tgz \
    -f docker/horovod/Dockerfile-sm80 .
    # --build-arg PYTORCH_LIGHTNING_VERSION=1.5.9 \
