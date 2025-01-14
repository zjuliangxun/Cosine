# This is only for GPU Arch < sm_80, like 3090, TITAN RTX(excluding 4090)
FROM dockerpull.org/horovod/horovod:master
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libglib2.0-0 \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 vulkan-tools \
 mesa-vulkan-drivers \
 pigz \
 libegl1 \
 git-lfs \
 tmux \
 sudo \
 zsh \
 gpustat \
&& rm -rf /var/lib/apt/lists/*

RUN pip install torch_geometric pillow imageio ninja scipy  gym \
    hydra-core omegaconf wandb pyrootutils termcolor ipdb pycollada pyglet\
    pydelatin opencv-python rl-games==1.1.4 pyfqmr -i https://mirrors.zju.edu.cn/pypi/web/simple

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# isaacgym download
WORKDIR /
RUN wget https://developer.nvidia.com/isaac-gym-preview-4 \
    && tar -zxf isaac-gym-preview-4 \
    && rm -rf isaac-gym-preview-4 \
    && cd ./isaacgym/python && pip install -e .