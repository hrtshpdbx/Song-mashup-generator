#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

sudo apt install -y ffmpeg
sudo apt install -y rubberband-cli

pip install -e .
