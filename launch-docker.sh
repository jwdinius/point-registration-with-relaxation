docker run -it --rm \
    -v $(pwd):/registration \
    --name register-c \
    --net host \
    --privileged \
    --runtime=nvidia \
    register
