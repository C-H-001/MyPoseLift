#!/bin/bash
set -e
# conda shell hook (needed for conda activate in non-interactive shells)
source /home/wsco/anaconda3/etc/profile.d/conda.sh
conda create -n ch_pose python=3.10 -y
conda activate ch_pose
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy opencv-python matplotlib tqdm tensorboard pandas scikit-learn
