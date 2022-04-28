# Layer-wise training on MNIST with PyTorch

This example utilizes FLoES's support for layer-wise training to train a single
layer on each of the client nodes, to collaboratively train the full network.

This is also sometimes called "progressive" training.

In this experiment, the model is a simple 3-layer model. The three clients each
train one of the layers. Each client has access to and trains on the full MNIST
training dataset.
