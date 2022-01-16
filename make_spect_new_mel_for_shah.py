import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
import argparse
import librosa
import pyloudnorm as pyln

parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', type=str, default='./wavs')
parser.add_argument('--target-dir', type=str, default='./spmel')
args = parser.parse_args()

# audio file directory
rootDir = args.root_dir
# spectrogram directory
targetDir = args.target_dir


def melspectrogram(
    wav,
    sr=16000,
    hop_length=200,
    win_length=800,
    n_fft=2048,
    n_mels=128,
    fmin=50,
    preemph=0.97,
    top_db=80,
    ref_db=20,
):
    mel = librosa.feature.melspectrogram(
        librosa.effects.preemphasis(wav, coef=preemph),
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        norm=1,
        power=1,
    )
    logmel = librosa.amplitude_to_db(mel, top_db=None) - ref_db
    logmel = np.maximum(logmel, -top_db)
    return logmel / top_db

# dirName, subdirList, _ = next(os.walk(rootDir))
subdirList = os.listdir(rootDir)
print('Found directory: %s' % rootDir)

for subdir in sorted(subdirList):
    print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    tmp_list = os.listdir(os.path.join(rootDir,subdir))
    for tmp_folder in tmp_list:
        tmp_list2 = os.listdir(os.path.join(rootDir,subdir, tmp_folder))
        for tmp_folder2 in tmp_list2:
            print('\t', tmp_folder,tmp_folder2)
            fileList = os.listdir(os.path.join(rootDir, subdir,tmp_folder, tmp_folder2))
            for fileName in sorted(fileList):
                print('\t\t', fileName)
                meter = pyln.Meter(16000)
                wav, _ = librosa.load(os.path.join(rootDir,subdir,tmp_folder, tmp_folder2,fileName), sr=16000)
                loudness = meter.integrated_loudness(wav)
                wav = pyln.normalize.loudness(wav, loudness, -24)
                peak = np.abs(wav).max()
                if peak >= 1:
                    wav = wav / peak * 0.999

                mel = melspectrogram(wav, n_mels=80) 
                mel = np.transpose(mel, (1, 0))
                # save spect    
                np.save(os.path.join(targetDir, subdir, f"{tmp_folder}_{tmp_folder2}_" + fileName[:-4]),
                        mel, allow_pickle=False)    
print('END')
        
