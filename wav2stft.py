import os
import numpy as np
import argparse
import sys
from scipy import signal
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile

#Usage: See run_wav2stft.sh


def main():
    parser = argparse.ArgumentParser(description='Compute magnitude spectrograms from WAV inputs')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1,
        help='GPU ID (negative value indicates CPU)')
    parser.add_argument(
        '--inpath', '-i', type=str,
        help='path for wav data')
    parser.add_argument(
        '--flen', '-l', type=int, default=2048,
        help='Frame length of STFT')
    parser.add_argument(
        '--fshift', '-s', type=float, default=1024, 
        help='Frame shift in ms of STFT')
    parser.add_argument(
        '--datpath', '-d', type=str, default='./dat/default/', 
        help='path for output dat files')
    parser.add_argument(
        '--figpath', '-f', type=str, default='./fig/default/', 
        help='path for output png files')
    parser.add_argument(
        '--prefix', '-p', type=str, default='default', 
        help='prefix for stat files')
    args = parser.parse_args()

    fshift = args.fshift
    flen = args.flen
        
    inpath = args.inpath
    datpath = args.datpath
    figpath = args.figpath
    prefix = args.prefix
    
    def mynum2str(i):
        if i < 10:
            return '000'+str(i)
        elif i < 100:
            return '00'+str(i)
        elif i < 1000:
            return '0'+str(i)
        else:
            return str(i)

    def wav2stft(wpath, dpath, fpath, flen, fshift, fs_resample):
        wdir = sorted(os.listdir(wpath))
        i = 0
        for fname in wdir:
            path_fname = wpath+fname
            print(path_fname)
            fs, data = wavfile.read(path_fname)    
            ddim = data.ndim
            data = data.reshape(ddim, -1)
            x = np.mean(data, axis=0)
            x = x.flatten()
            J = int(np.ceil(x.shape[0]*fs_resample/fs))
            y = signal.resample(x, J)
            nchannels = data.shape[0]
            nsamples = len(y)
            print('Channel num: ', nchannels)
            print('Sampling rate: ', fs_resample)
            print('Sample num: ', nsamples)
            print('Frame length: {}(ms)'.format(flen*1000.0/fs_resample))
            print('Frame shift: {}(ms)'.format(fshift*1000.0/fs_resample))

            f, t, Y = signal.stft(y, fs_resample, nperseg=flen, nfft=flen, noverlap=flen-fshift, window='hamming')
            freqnum, framenum = Y.shape

            np.save(dpath+'cspec/'+mynum2str(i)+'cspec.npy', Y)
            
            #Y_real = Y.real
            #Y_imag = Y.imag
            #Y_realimag = Y_real[np.newaxis,:,:]
            #Y_realimag = np.append(Y_realimag, Y_imag[np.newaxis,:,:], axis=0)
            #np.save(dpath+'cspec/'+mynum2str(i)+'cspec.npy', Y_realimag)
            
            plt.clf()
            plt.pcolormesh(t, f, abs(Y)**0.3)
            plt.title('Magnitude spectrogram')
            plt.ylabel('Frequency')
            plt.xlabel('Time')
            plt.savefig(fpath+'mspec/'+mynum2str(i)+'mspec.png')
                        
            i += 1

    if not os.path.exists(args.datpath):
        os.makedirs(args.datpath)
    if not os.path.exists(args.datpath+'mspec/'):
        os.makedirs(args.datpath+'mspec/')
    if not os.path.exists(args.datpath+'cspec/'):
        os.makedirs(args.datpath+'cspec/')
    if not os.path.exists(args.figpath):
        os.makedirs(args.figpath)
    if not os.path.exists(args.figpath+'mspec/'):
        os.makedirs(args.figpath+'mspec/')
   
    wav2stft(inpath, datpath, figpath, flen, fshift, 16000)
    compstat(datpath, prefix)
    

def compstat(dpath,prefix):
    DataDir = sorted(os.listdir(dpath+'cspec/'))
    Ndata = len(DataDir)
    i = 0
    Yconcat = np.array([])
    Xconcat = np.array([])
    for fname in DataDir:

        cspec_path_fname = dpath + 'cspec/' + fname
        
        print(cspec_path_fname)
        X = np.load(cspec_path_fname)
        freqnum, framenum = X.shape
        if i==0:
            Xconcat = X
        else:
            Xconcat = np.append(Xconcat,X,axis=1)
        i += 1
    n_t = Xconcat.shape[1]

    Xconcat_real = Xconcat.real[1:freqnum,:]
    Xconcat_imag = Xconcat.imag[1:freqnum,:]
    Y_gm_real = Xconcat_real.mean(axis=(0,1))
    Y_gm_imag = Xconcat_imag.mean(axis=(0,1))

    Yconcat = Xconcat_real[np.newaxis,:,:]
    Yconcat = np.append(Yconcat,Xconcat_imag[np.newaxis,:,:],axis=0)
    #Yconcat.shape: 2, freqnum-1, framenum

    Y_abs1 = abs(Xconcat[np.newaxis,1:freqnum,:])
    Y_abs = np.linalg.norm(Yconcat,axis=0,keepdims=True)
    error = np.sum(np.power(Y_abs1-Y_abs,2))
    print('Error: '+str(error))
    #Y_abs.shape: 1, freqnum-1, framenum
    Y_gv = np.mean(np.power(Y_abs, 2),axis=(0,1,2),keepdims=True)
    Y_gs = np.sqrt(Y_gv)

    #Y_gs = Yconcat.std(axis=(0,1,2))
    #Y_gmgs = np.array([Y_gm_real,Y_gm_imag,Y_gs])
    #Y_gm = Yconcat.mean(axis=(0,1,2), keepdims=True)
    #Y_gs = Yconcat.std(axis=(0,1,2), keepdims=True)
    Y_gvgs = Y_gv
    Y_gvgs = np.append(Y_gvgs,Y_gs,axis=0)
    np.save(dpath+prefix+'cspecstat.npy',Y_gvgs)


if __name__ == '__main__':
    main()
