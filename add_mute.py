import os
import soundfile as sf
import math
import librosa
import random
import numpy as np
import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--clean_file', type=str, default='', help='file of clean wave')
	parser.add_argument('--mute_file', type=str, default='', help='file of mute wave')
	args = parser.parse_args()
	return args

def add_noise(audio_path, out_path, sr=16000):
    src, sr = librosa.core.load(audio_path, sr=sr)

    # set random insert position
    l1 = random.randint(1, len(src))
    l2 = random.randint(1, len(src))


    # insert mute section
    insert = np.zeros(round(len(src)/2), dtype=float, order='C')
    src = np.insert(src, l1, insert)
    src = np.insert(src, l2, insert)


    sf.write(out_path, src, sr)


def main():
    args = get_args()
    #parameter and path settings
    path = {'clean_file': args.clean_file, 'mute_file': args.mute_file}


    clean_root = args.clean_file
    clean_dir_names = sorted(os.listdir(clean_root))
    mix_root = args.mute_file
    os.makedirs(mix_root, exist_ok=True)

    for idx, f in enumerate(clean_dir_names):
        clean_dir = os.path.join(clean_root, f)
        save_dir = mix_root
        os.makedirs(save_dir, exist_ok=True)


        mix_name = f[0:-4] + ' noisy.wav'
        add_noise(clean_dir, os.path.join(save_dir, mix_name))




if __name__=="__main__":
    main()

