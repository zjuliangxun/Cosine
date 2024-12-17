FROM dockerpull.org/horovod/horovod:master
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
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
 && rm -rf /var/lib/apt/lists/*

RUN pip install torch_geometric pillow imageio ninja scipy  gym \
    hydra-core omegaconf wandb pyrootutils termcolor ipdb pycollada pyglet\
    pydelatin opencv-python rl-games pyfqmr -i https://mirrors.zju.edu.cn/pypi/web/simple

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# isaacgym download
WORKDIR /
RUN wget https://developer.download.nvidia.com/isaac/secure/Isaac_Gym/Preview_4/IsaacGym_Preview_4_Package.tar.gz?LHzYYHubxlknOinUOQ2LOWPWZJ003yFroRdwfhsH7nmQRJ1c6gDUfZ4l4uApItPjLZAfyvKlCgyJdjBTQ-efASZBXlzpKeAG_7BOUzZN6pS7OdXthsscUyrvyqkaJq4kMmgEFmcuisAtnlZ9oTT53M2iYpkTRPqFskr7Nx1K0nhr&t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1naXRodWIuY29tL3JvYm90bGVhcm5pbmcxMjMvYXdlc29tZS1pc2FhYy1neW0/dGFiPXJlYWRtZS1vdi1maWxlIn0= \
    && mkdir isaacgym && cd isaacgym
    && tar -zxf IsaacGym_Preview_4_Package.tar.gz -c isaacgym
    && cd ./isaacgym/python && pip install -e .
    && rm -rf IsaacGym_Preview_4_Package.tar.gz

# isaacgym local install
# RUN mkdir /isaacgym && cd /
# COPY . ./isaacgym
# RUN cd ./isaacgym/python && pip install -e .