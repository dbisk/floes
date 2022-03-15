# Raspberry Pi Setup

The main thing for the Raspberry Pi setup is that a 64-bit operating system is
required. The best (and tested) option is Raspberry Pi OS Bullseye, 64-bit,
available through the official Raspberry Pi Imager.

After setting up the Raspberry Pi through the normal steps, then simply `scp`
or `wget` the `floes` wheel and the `requirements.sh` in this folder to the
home directory of the Pi. Then simply execute the script:
```
bash requirements.sh
```
which should install everything necessary to run FLoES on the Raspberry Pi.

