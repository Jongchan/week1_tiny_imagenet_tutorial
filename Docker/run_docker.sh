USER_HOME=/home/$USER
docker run \
    -d \
    --ipc=host \
    -it \
    -v ${USER_HOME}:${USER_HOME} \
    --privileged=true \
    --network=host \
    $USER/pytorch:1.6.0-cuda10.1-cudnn7-devel
    bash -c 'cd "${USER_HOME}"; exec "${SHELL:-sh}"'
