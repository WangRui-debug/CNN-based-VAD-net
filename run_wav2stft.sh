#!/bin/sh

# Feature extraction from training data

python wav2stft.py \
	-i ./mix/01b/ \
       -d ./npy_01b_noisy/wsj_training/01b_noisy/ \
       -f ./fig_01b_noisy/wsj_training/01b_noisy/ \
       -l 2048 -s 1024 -p train_
