"""
utility.py - tiny utility functions for manipulating how the dataset is laid
out in storage. Not used in the experiment whatsoever.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import os


def split(a, n):
    """Returns a generator with the splits of a given list a."""
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def separate_into_n_clients(data_root_path: str, n: int, suffix: str = 'train'):
    all_ids = os.listdir(data_root_path)
    split_ids = [i for i in split(all_ids, n)]

    for client_idx, client_ids in enumerate(split_ids):
        os.makedirs(os.path.join(data_root_path, f"client{client_idx}", suffix))
        for speaker_id in client_ids:
            os.rename(
                os.path.join(data_root_path, speaker_id),
                os.path.join(
                    data_root_path, f"client{client_idx}", suffix, speaker_id
                )
            )


def consolidate_all_clients(data_root_path: str, suffix: str = 'train'):
    all_clients = os.listdir(data_root_path)
    for client in all_clients:
        speaker_ids = os.listdir(os.path.join(data_root_path, client, suffix))
        for speaker_id in speaker_ids:
            os.rename(
                os.path.join(data_root_path, client, suffix, speaker_id),
                os.path.join(data_root_path, speaker_id)
            )
        os.removedirs(os.path.join(data_root_path, client, suffix))

