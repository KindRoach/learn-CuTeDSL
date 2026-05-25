FROM nvcr.io/nvidia/pytorch:26.04-py3

# A cute Layout visualization tool
RUN pip install tensor-layouts[viz]

# config user and bash completion
RUN apt-get update && apt-get install -y sudo bash-completion \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu
