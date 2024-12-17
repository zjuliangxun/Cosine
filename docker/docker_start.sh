#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
DOCKER_HOME="/home/$USER"

IMG="isaacgym:v0"  

LOCAL_DIR=$(pwd)
if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_isaacgym_docker"
fi

USER_ID=$(id -u)
GRP=$(id -g -n)
GRP_ID=$(id -g)

local_volumes() {
  volumes="-v $LOCAL_DIR:$LOCAL_DIR \
           -v $HOME/.ssh:${DOCKER_HOME}/.ssh \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		   -v /media:/media \
		   -v /etc/localtime:/etc/localtime:ro \
		   -v /nfs:/nfs \
       -v /mnt:/mnt \
		   -v /ssd:/ssd \
       -v /raid:/raid \
		   -v ${HOME}/.torch:${DOCKER_HOME}/.torch \
		   -v ${HOME}/.cache:${DOCKER_HOME}/.cache \
		   -v /data:/data"
  echo "${volumes}"
}


add_user() {
    add_script="addgroup --gid ${GRP_ID} ${GRP} && \
    adduser --disabled-password --gecos '' ${USER} \
    --uid ${USER_ID} --gid ${GRP_ID} 2>/dev/null && \
    usermod -aG sudo ${USER} && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    cp -r /etc/skel/. /home/${USER} && \
    chsh -s /usr/bin/zsh ${USER} && \
    chown -R ${USER}:${GRP} '/home/${USER}'"
  echo "${add_script}"
}


main(){
    docker ps -a --format "{{.Names}}" | grep "${DOCKER_NAME}" 1>/dev/null
    if [ $? == 0 ]; then
        docker stop ${DOCKER_NAME} 1>/dev/null
        docker rm -f ${DOCKER_NAME} 1>/dev/null
    fi

    local display="${DISPLAY:-:0}"

    DOCKER_CMD="docker"
    GPU_CONFIG="--gpus all --runtime=nvidia"


    LOCAL_HOST=`hostname`
    eval ${DOCKER_CMD} run -it \
        -d \
        $GPU_CONFIG \
        --name ${DOCKER_NAME}\
        -e DISPLAY=$display \
        -e DOCKER_USER=$USER \
        -e USER=$USER \
        -e DOCKER_USER_ID=$USER_ID \
        -e DOCKER_GRP=$GRP \
        -e DOCKER_GRP_ID=$GRP_ID \
        -e DOCKER_HOME=$DOCKER_HOME \
        -e SSH_AUTH_SOCK=/tmp/.ssh-agent-$USER/agent.sock \
        $(local_volumes) \
        -p $RANDOM:22 \
        -p $RANDOM:8501 \
        -p $RANDOM:6006 \
        -w $LOCAL_DIR \
        --add-host ${HOSTNAME}:127.0.0.1 \
        --add-host ${LOCAL_HOST}:127.0.0.1 \
        --hostname ${HOSTNAME} \
        --shm-size 1024G \
        $IMG   
    docker exec ${DOCKER_NAME} service ssh start
    docker exec ${DOCKER_NAME} echo "$(add_user)"
    docker exec ${DOCKER_NAME} bash -c "$(add_user)"
}

main
