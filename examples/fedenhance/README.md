# Unsupervised Audio Enhancement with FedEnhance

This example implements the unsupervised setup described in the paper
"Separate but Together: Unsupervised Federated Learning for Speech
Enhancement from Non-IID Data"[^1]. This example uses the LibriFSD50K dataset
described in [1]. 


[^1]: Efthymios Tzinis, Jonah Casebeer, Zhepei Wang, and Paris Smaragdis.
Separate but Together: Unsupervised Federated Learning for Speech Enhancement
from Non-IID Data. WASPAA 2021. https://arxiv.org/abs/2105.04727. 

## Server Setup

The entry points to this example are through `sudo_client.py` and
`sudo_server.py`. For default settings replicating the setup described in [1]:
```bash
python sudo_server.py --address ADDRESS_TO_EXPOSE --rounds NUMBER_OF_FEDERATED_ROUNDS \
    --min_clients MINIMUM_CLIENTS_ALLOWED_TO_CONNECT \
    --max_clients MAXIMUM_CLIENTS_ALLOWED_TO_CONNECT
```

## Client Setup

The client script can be run on clients such as PCs, Raspberry Pis, and NVIDIA
Jetson Nanos. For default settings:
```bash
python sudo_client.py --data_dir PATH_TO_LIBRIFSD50K_DATA --address ADDRESS_OF_SERVER
```
To run all clients on a single machine, a convenience script
`run_on_one_machine.bash` is available. It automatically parallelizes the
clients to different GPUS. Note that each client with default settings uses
around 2.2 GB of GPU memory. Usage:
```bash
./run_on_one_machine.bash NUM_CLIENTS_TO_START LIBRIFSD50K_PATH --address SERVER_ADDRESS
```
Any additional arguments to pass into `sudo_client.py` can be appended the end
the script arguments.

## Notes

- If simulating the federated setup by running both the server and all clients
on a single machine, the `--address` argument can be omitted.
- The data directory is to the _root_ of the LibriFSD50K directory. The
subfolders expected are `test`, `train`, and `val`. Each client will train off
the entirely of the `train` folder in the path that it is given. To reproduce
the results in [1], the full LibriFSD50K dataset will need to be separated into
separate folders.

