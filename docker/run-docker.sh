# NOTE: run this from dir above docker!
# usage ./docker/run-docker.sh {--runtime=nvidia}
# ** only pass the --runtime=nvidia arg if you have the docker nvidia runtime setup correctly! **
docker run -it --rm \
    -v $(pwd):/home/relax/registration \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    --name qap-register-c \
    --net host \
    --privileged \
    $1 \
    jdinius/qap-register:latest 
