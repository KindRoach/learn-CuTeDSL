FROM nvcr.io/nvidia/pytorch:26.06-py3

# config user and bash completion
RUN apt-get update && apt-get install -y sudo bash-completion \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# A cute Layout visualization tool
RUN pip install tensor-layouts[viz]
