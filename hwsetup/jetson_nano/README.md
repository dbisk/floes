# NVIDIA Jetson Nano Setup

The Jetson Nanos have some specific library versions that are separate from
those found in PyPI. This document describes how to set up a Jetson Nano for
general training with PyTorch as well as for FLoES.

## Jetson OS Image

The Jetson OS is called JetPack. Initial setup for the Jetson Nano can be found
on [NVIDIA's website](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).

Once the Jetson Nano is set up with the provided image, we can install our
deep learning libraries.

Note: you may want to update your system with
```
$ sudo apt update
$ sudo apt upgrade
```

## Installing PyTorch

PyTorch can be installed via wheel from the
[Jetson Zoo](https://elinux.org/Jetson_Zoo), under the instructions for
PyTorch. FLoES requires Python 3.5 or higher and PyTorch 1.8 or higher. After
downloading the wheel from the Jetson Zoo, some PyTorch dependencies are
required.
```
$ sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev python3-pip
$ sudo -H pip3 install Cython
$ sudo -H pip3 install numpy==1.19.4
$ sudo -H pip3 install <name of torch wheel download from Jetson Zoo>
```

## Installing FLoES

Finally, we can install FLoES and its dependencies. With the FLoES wheel
downloaded, simply run:
```
$ sudo -H pip3 install <name of FLoES wheel>
```
 
