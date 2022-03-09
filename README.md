# CNN-based-VAD-net
This is a script to generate the noisy training data and binary label for the training of VAD net

First, please install py-webrtcvad by：
pip install webrtcvad

For detailed information, please check follwing instrucion: https://github.com/wiseman/py-webrtcvad

1. The generation of noisy data
   addnoise.py is used to insert mute into a wav file and add noise with desired SNR.
   useage: python3 ./addnoise.py --clean_file ./clean --noise_file ./noise.wav --mix_file ./mix --snr 0
   
2. The gerneration of corresponding label by using webrtcvad. The sensitivity of VAD is from 0 to 3. 
   python3 ./label_generation.py --mix_file ./mix --label_file ./label --sensitivity 3 --batch_size 3

The code of training is in progress. 
2022.3.9
