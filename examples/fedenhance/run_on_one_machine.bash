#!/bin/bash

echo "Running FedEnhance on one machine for $1 clients."
echo "Data directory given as: $2"
echo "Assuming directory structure with each client's training set in \
the \"client{i}\" folder in the data directory."
echo "Additional client parameters given: ${@:3}"

TOTAL_CUDA_DEVICES=$(nvidia-smi -L | wc -l)
echo "Found $TOTAL_CUDA_DEVICES CUDA GPUs."

trap 'kill 0' SIGINT
for (( i = 0; i < $1; i++ )); do
    CUDA_DEVICE=$(( $1 %  $TOTAL_CUDA_DEVICES ))
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE poetry run python sudo_client.py --data_dir ${2%/}/client$i ${@:3} &

    sleep 1
done

echo "$1 clients started. Awaiting their completion."
wait
