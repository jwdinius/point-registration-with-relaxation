FROM jdinius/nmsac-deps

ARG username=relax
ENV USER=$username

# setup user env at the end
# -m option creates a fake writable home folder
RUN adduser --disabled-password --gecos '' $username
RUN adduser $username sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $username
WORKDIR /home/$username
