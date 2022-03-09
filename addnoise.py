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
	parser.add_argument('--noise_file', type=str, default='', help='file of noise wave')
	parser.add_argument('--mix_file', type=str, default='', help='file of mix wave')
	parser.add_argument('--snr', type=float, default='', help='SNR')
	args = parser.parse_args()
	return args

def add_noise(audio_path, noise_path,out_path, SNR, sr=16000):
    src, sr = librosa.core.load(audio_path, sr=sr)

    # set random insert position
    l1 = random.randint(1, len(src))
    l2 = random.randint(1, len(src))


    # insert mute section
    insert = np.zeros(round(len(src)/2), dtype=float, order='C')
    src = np.insert(src, l1, insert)
    src = np.insert(src, l2, insert)

    random_values = np.random.rand(len(src))

    #Calculation of SNR
    Ps = np.sum(src ** 2) / len(src)
    Pn1 = np.sum(random_values ** 2) / len(random_values)
    k=math.sqrt(Ps/(10**(SNR/10)*Pn1))
    random_values_we_need=random_values*k
    Pn=np.sum(random_values_we_need**2)/len(random_values_we_need)
    snr=10*math.log10(Ps/Pn)
    print("The snr is ：",snr)

    #add noise
    sf.write(noise_path,random_values_we_need, sr)
    outdata=src+random_values_we_need
    sf.write(out_path, outdata, sr)


def main():
    args = get_args()
    #parameter and path settings
    path = {'clean_file': args.clean_file, 'noise_file': args.noise_file, 'mix_file': args.mix_file}
    condition = {'snr': args.snr}

    clean_root = args.clean_file
    clean_dir_names = sorted(os.listdir(clean_root))
    mix_root = args.mix_file
    os.makedirs(mix_root, exist_ok=True)

    for idx, f in enumerate(clean_dir_names):
        clean_dir = os.path.join(clean_root, f)
        save_dir = mix_root
        os.makedirs(save_dir, exist_ok=True)

        noisename= args.noise_file
        SNR = args.snr
        mix_name = f[0:-4] + ' with noise.wav'
        add_noise(clean_dir, noisename, os.path.join(save_dir, mix_name), SNR)




if __name__=="__main__":
    main()

