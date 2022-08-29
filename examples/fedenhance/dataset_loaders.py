"""
dataset_loaders.py - 
"""

import os
import chunked_libri_fsd

CHUNK_LIBRI_FS50K = '/data/ChunkedLibriFS50K'

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def enhancement_federated_setup(hparams):
    # Create all generators
    available_speaker_ids = os.listdir(os.path.join(CHUNK_LIBRI_FS50K, 'train'))
    # Split the available speaker ids to the individual nodes.
    splitted_ids = list(split(list(range(len(available_speaker_ids))),
                              hparams['n_fed_nodes']))

    # The first X nodes are always supervised, declared by the parameter.
    federated_generators_list = []
    num_supervised_nodes = int(
        hparams['p_supervised'] * len(splitted_ids))
    for j, ids in enumerate(splitted_ids):
        is_supervised = j < num_supervised_nodes

        data_loader = chunked_libri_fsd.Dataset(
            root_dirpath=CHUNK_LIBRI_FS50K,
            speaker_ids=ids,
            available_speech_percentage=hparams['available_speech_percentage'],
            split='train', sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=True)
        train_generator = data_loader.get_generator(
            batch_size=hparams['batch_size'],
            num_workers=hparams['n_jobs'])

        federated_generators_list.append(
            {'is_supervised': is_supervised,
             'node_id': j,
             'speaker_ids': ids,
             'train_generator': train_generator})

    val_generators = {}
    for data_split in ['val', 'test']:
        available_speech_percentage = 0.5
        data_loader = chunked_libri_fsd.Dataset(
            root_dirpath=CHUNK_LIBRI_FS50K,
            speaker_ids=[],
            available_speech_percentage=available_speech_percentage,
            split=data_split, sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=data_split=='train')
        val_generators[data_split] = data_loader.get_generator(
            batch_size=hparams['batch_size'],
            num_workers=hparams['n_jobs'])
    return federated_generators_list, val_generators
