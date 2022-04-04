"""
haiku_mnist_client.py - Implementation of a Haiku/JAX client using FLoES.
Trains a simple network on the MNIST image classification dataset. 

This example uses the same split and model as the PyTorch example, and thus the
expected results are the same.

Note that this example also requires `torch` and `torchvision`, in order to
perform dataloading. This part is entirely unrelated to FLoES.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from typing import List, Tuple, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import floes.client
from floes.core import FloesParameters

from haiku_mnist_common import net_fn, numpy_collate, TorchToJAXTransform, Batch


class MNISTClient(floes.client.Client):

    def __init__(self, net, params: hk.Params):
        super().__init__()
        self.net = net
        self.hk_params = params
        self._create_keymap(self.hk_params)

    def _create_keymap(self, params: hk.Params):
        self._keymap: Dict[str, Tuple[str, str]] = {}
        for top_key, lower_dict in params.items():
            for lower_key in lower_dict.keys():
                self._keymap[f'{top_key}:{lower_key}'] = (top_key, lower_key)

    def get_parameters(self) -> FloesParameters:
        params_dict = {}
        for top_key, lower_dict in self.hk_params.items():
            for lower_key, weights in lower_dict.items():
                params_dict[f'{top_key}:{lower_key}'] = np.array(weights)
        return FloesParameters(params_dict)

    def set_parameters(self, params: FloesParameters):
        for layer, weights in params.items():
            top_key, lower_key = self._keymap[layer]
            self.hk_params[top_key][lower_key] = jnp.array(weights)
    
    def set_train_metadata(self, train_dataloader, optimizer):
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
    
    def train(self):
        net = self.net
        opt = self.optimizer
        opt_state = self.optimizer.init(self.hk_params)

        # Training loss (cross-entropy).
        def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
            """Compute the loss of the network, including L2."""
            logits = net.apply(params, batch)
            labels = jax.nn.one_hot(batch["label"], 10)

            l2 = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
            softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            return softmax_xent + 1e-4 * l2
        
        # Evaluation metric (classification accuracy).
        @jax.jit
        def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
            predictions = net.apply(params, batch)
            return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

        @jax.jit
        def update(
            params: hk.Params,
            opt_state: optax.OptState,
            batch: Batch,
        ) -> Tuple[hk.Params, optax.OptState]:
            """Learning rule (stochastic gradient descent)."""
            grads = jax.grad(loss)(params, batch)
            updates, opt_state = opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state
        
        # train loop
        train_dataloader = self.train_dataloader
        dataloader_length = len(train_dataloader)
        with tqdm(train_dataloader, unit='batches') as tbatch:
            count = 0 # only needed for progress bar
            for X, y in tbatch:
                batch = {"image": X, "label": y}

                # do SGD on a batch of training samples
                self.hk_params, opt_state = update(
                    self.hk_params,
                    opt_state,
                    batch
                )

                # periodically assess the accuracy
                if count % (int(dataloader_length / 5)) == 0:
                    acc = accuracy(self.hk_params, batch)
                    count = 0
                    tbatch.set_postfix_str(f'Training Acc: {acc:.3f}')
                count += 1


def evaluate_model(net, hk_params: hk.Params) -> Dict:
    # load and prepare MNIST test dataset
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=TorchToJAXTransform()
    )

    test_dataloader = DataLoader(test_data, batch_size=10, collate_fn=numpy_collate, shuffle=False)

    # define the loss function
    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        logits = net.apply(params, batch)
        labels = jax.nn.one_hot(batch["label"], 10)

        l2 = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        return softmax_xent + 1e-4 * l2
    
    @jax.jit
    def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
        predictions = net.apply(params, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])
    
    # perform the evaluation loop
    total_batches = 0
    acc = 0
    with tqdm(test_dataloader, unit='batches') as tbatch:
        for X, y in tbatch:
            total_batches += 1
            batch = {"image": X, "label": y}
            acc += accuracy(hk_params, batch)
    acc = acc / total_batches
    return {'accuracy': 100. * acc}


def main():
    # Make the network and optimiser.
    net = hk.without_apply_rng(hk.transform(net_fn))
    opt = optax.sgd(1e-3)

    # make datasets
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=TorchToJAXTransform()
    )

    # only keep a certain subset of the training dataset for this client
    fraction = 0.1
    idxs = np.random.randint(0, len(train_data), size=int(len(train_data) * fraction))
    train_data = Subset(train_data, idxs)

    # define dataloaders for the datasets
    train_dataloader = DataLoader(
        train_data,
        collate_fn=numpy_collate,
        batch_size=1,
        shuffle=True,
        drop_last=True,
    )

    # initialize the network and optimizer; note we draw an input to get shapes
    batch = next(iter(train_dataloader))
    batch = {"image": batch[0], "label": batch[1]}
    params = net.init(jax.random.PRNGKey(42), batch)

    # create the floes client
    client = MNISTClient(net, params)
    client.set_train_metadata(train_dataloader, opt)

    # do one round of local training to get a baseline
    client.train()
    print(
        "Initial Accuracy:",
        evaluate_model(net, client.hk_params)['accuracy']
    )

    # set address information
    address = 'localhost:50051'

    # start the GRPC connection and client loop
    # this will continue until server indicates it is done
    print("Awaiting for signal from server to begin")
    trained_client = floes.client.start_client(client, address)
    print("Server indicates training done. Disconnected from server.")

    # evaluate the final model
    print("Evaluating final model on local test set...")
    print(
        "Final Accuracy:",
        evaluate_model(net, trained_client.hk_params)['accuracy']
    )


if __name__ == '__main__':
    main()
