#!/bin/sh

# quick setup script that installs needed dependencies for torch and floes
sudo apt update
sudo apt upgrade -y
sudo apt install -y git vim
sudo apt install -y python3-pip python3-venv libopenblas-dev libopenmpi-dev libomp-dev

# install the virtual environment for torch
python3 -m venv .venv
source .venv/bin/activate
pip install setuptools==58.3.0
pip install Cython
pip install torch

# install floes (which installs grpc)
pip install floes-0.0.3-py3-none-any.whl
