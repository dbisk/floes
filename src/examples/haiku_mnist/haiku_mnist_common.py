
from typing import Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Batch = Mapping[str, np.ndarray]


class MNISTModel(hk.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = hk.Conv2D(32, kernel_shape=(3, 3))
        self.relu = jax.nn.relu
        self.flatten = hk.Flatten()
        self.fc1 = hk.Linear(128)
        self.fc2 = hk.Linear(10)

    def __call__(self, x):
        out = self.relu(self.conv1(x))
        out = self.fc1(self.flatten(out))
        out = self.fc2(out)
        return out


def net_fn(batch: Batch) -> jnp.ndarray:
    x = batch['image'].astype(jnp.float32) / 255.
    model = MNISTModel()
    return model(x)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class TorchToJAXTransform(object):

    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32)