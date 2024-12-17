#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# should match env name from YAML
ENV_NAME=isaacgym

pushd "${ROOT_DIR}/python"

    # setup conda
    CONDA_DIR="$(conda info --base)"
    source "${CONDA_DIR}/etc/profile.d/conda.sh"

    # deactivate the env, if it is active
    ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
    if [ "${ENV_NAME}" = "${ACTIVE_ENV_NAME}" ]; then
        conda deactivate
    fi

    # !!! this removes existing version of the env
    conda remove -y -n "${ENV_NAME}" --all

    # conda env create -f ./rlgpu_conda_env.yml
    # * create env only
    conda create -n "${ENV_NAME}" python=3.8

    if [ $? -ne 0 ]; then
        echo "*** Failed to create env"
        exit 1
    fi

    # activate env
    conda activate "${ENV_NAME}"
    if [ $? -ne 0 ]; then
        echo "*** Failed to activate env"
        exit 1
    fi

    # double check that the correct env is active
    ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
    if [ "${ENV_NAME}" != "${ACTIVE_ENV_NAME}" ]; then
        echo "*** Env is not active, aborting"
        exit 1
    fi
    
    conda install pytorch torchvision pytorch-cuda=11.7 pyg=*=*cu* horovod -c pytorch -c nvidia -c pyg -c conda-forge
    conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
    conda install numpy jupyter ipykernel pillow imageio ninja pyyaml scipy  gym \
        tensorboard hydra-core omegaconf wandb pyrootutils termcolor ipdb setuptools=69.5.1 \
        pydelatin opencv -c conda-forge
    pip install rl-games pyfqmr
    # conda install -c conda-forge robot_descriptions yourdfpy pycollada pyglet
    
    # install isaacgym package
    pip install -e .

popd

echo "SUCCESS"