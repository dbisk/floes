#!/bin/sh

# quick setup script that installs needed dependencies for torch and floes
sudo apt update
sudo apt upgrade -y
sudo apt install -y git vim
sudo apt install -y python3-pip python3-venv libopenblas-dev libopenmpi-dev libomp-dev

# install torch (note: installs system wide. only do this on edge platforms.)
sudo -H pip3 install setuptools==58.3.0
sudo -H pip3 install Cython
sudo -H pip3 install torch

# install floes (which installs grpc)
sudo -H pip3 install floes-0.0.3-py3-none-any.whl
