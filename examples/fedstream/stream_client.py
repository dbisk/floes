"""
stream_client.py - the main client script for the FedStream example.

@author Dean Biskup
@email dbiskup2@illinois.edu
@org University of Illinois, Urbana-Champaign Audio Group
"""

import argparse

def main(args):
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        required=True,
        help="The address of the server to connect to."
    )
    parser.add_argument(
        "--audio_timelength",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=16000
    )
    
    args = parser.parse_args()
    main(args)
