"""
dataset_loaders.py - utility functions for creating the torch dataloaders for
the LibriFSD50K dataset.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import chunked_libri_fsd


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def client_loader(filepath: str, is_supervised: bool, hparams):
    """
    Creates the dataloader for a single client. Should be run on the client.

    Args:
        filepath: str
            The root path of the directory the dataset is stored in.
    Returns:
        Dataloader
            Torch dataloader generating the training samples.
    """
    data_loader = chunked_libri_fsd.Dataset(
        root_dirpath=filepath,
        available_speech_percentage=hparams['available_speech_percentage'],
        split='train',
        sample_rate=hparams['fs'],
        timelength=hparams['audio_timelength'],
        zero_pad=True,
        augment=True
    )
    train_generator = data_loader.get_generator(
        batch_size=hparams['batch_size'],
        num_workers=hparams['n_jobs']
    )

    return train_generator


def test_loader(filepath: str, hparams):
    """
    Generates the data loader for the test dataset. Used for both clients and
    the server

    Args:
        filepath: str
            The filepath to the root directory of the dataset.
    Returns:
        Dataloader
            Torch dataloader generating the test dataset.
    """
    data_loader = chunked_libri_fsd.Dataset(
        root_dirpath=filepath,
        available_speech_percentage=0.5,
        split='test',
        sample_rate=hparams['fs'],
        timelength=hparams['audio_timelength'],
        zero_pad=True,
        augment=False
    )
    test_generator = data_loader.get_generator(
        batch_size=hparams['batch_size'],
        num_workers=hparams['n_jobs']
    )
    return test_generator
