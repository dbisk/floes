"""
mnist_server.py - Server for the example using federated learning for MNIST. 
"""

import torch

import floe.server
import floe.strategy


class MNISTModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(32 * 26 * 26, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)


def main():
    address = '[::]:50051'

    # create the model
    # NOTE: this is jank
    model = MNISTModel()
    model = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # start the server
    # note: this never returns
    floe.server.start_server(
        model,
        address,
        3,
        floe.strategy.UnweightedFedAvg()
    )
    


if __name__ == '__main__':
    main()