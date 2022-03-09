import os
import collections
import contextlib
import sys
import wave
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import argparse

import webrtcvad

vad = webrtcvad.Vad()


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mix_file', type=str, default='', help='file of mix wave')
	parser.add_argument('--label_file', type=str, default='', help='file of label')
	parser.add_argument('--sensitivity', type=float, default='', help='aggressiveness of VAD')
	parser.add_argument('--batch_size', type=float, default='', help='batch size')
	args = parser.parse_args()
	return args

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

vad.set_mode(1)

# Run the VAD on 10 ms of silence. The result should be False.


def main():
    args = get_args()
    path = {'mix_file': args.mix_file, 'label_file': args.label_file}
    sensitivity = {'sensitivity': args.sensitivity}
    batch_size = {'batch_size': args.batch_size}

    label =np.array([])

    mix_root = args.mix_file
    mix_dir_names = sorted(os.listdir(mix_root))
    label_root = args.label_file
    os.makedirs(label_root, exist_ok=True)

    for idx, f in enumerate(mix_dir_names):
        mix_dir = os.path.join(mix_root, f)
        label_dir = label_root
        os.makedirs(label_dir, exist_ok=True)
        audio, sample_rate = read_wave(mix_dir)
        vad = webrtcvad.Vad(int(args.sensitivity))
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            if is_speech:
                label = np.append(label, 1.)
            else:
                label = np.append(label, 0.)

        label = np.atleast_2d(label)
        label = np.repeat(label[np.newaxis, :, :], args.batch_size, axis=0)

        label = torch.tensor(label)
        F.interpolate(label, size=None, scale_factor=4, mode='linear')

        label_name = f + '.npy'
        np.save(os.path.join(label_dir, label_name), label)
        print(label_name + 'was generated successfully!')


if __name__ == '__main__':
    main()