# NOTE: run this from dir above scripts!
docker run -it --rm \
    -v $(pwd):/home/relax/registration \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    --name register-nvidia-c \
    --net host \
    --privileged \
    --runtime=nvidia \
    register-nvidia
