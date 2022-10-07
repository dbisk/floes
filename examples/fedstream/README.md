# Unsupervised, streamed Audio Enhancement with Causal FedEnhance

This example implements an unsupervised setup in which the client nodes are
set up to perform fully online learning with incoming data samples. 

## Server Setup

The entry point to this example from the server side is through
`stream_server.py`. For default settings,
```bash
python stream_server.py --address ADDRESS_TO_EXPOSE --rounds NUMBER_OF_FEDERATED_ROUNDS \
    --min_clients MINIMUM_CLIENTS_ALLOWED_TO_CONNECT \
    --max_clients MAXIMUM_CLIENTS_ALLOWED_TO_CONNECT
```

## Client Setup

The client script specifically uses clients with the ReSpeaker microphone array
for Raspberry Pis. This is only tested on Raspberry Pis. For default settings,
```bash
python stream_client.py --address ADDRESS_OF_SERVER
```
