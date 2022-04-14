import argparse
import audiofile as af
import glob
import numpy as np
import os
import pandas as pd
import torch
import tqdm
import torchaudio
import torchlibrosa


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mel-spectrogram feature extraction')
    parser.add_argument(
        'root',
        help='Path to EXVO wav files'
    )
    parser.add_argument(
        'dest',
        help='Path to store features in'
    )
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    ref = 1.0
    amin = 1e-10
    top_db = None

    spectrogram = torchlibrosa.stft.Spectrogram(
        n_fft=512, 
        win_length=512, 
        hop_length=160
    )
    mel = torchlibrosa.stft.LogmelFilterBank(
        sr=16000, 
        fmin=50, 
        fmax=8000, 
        n_mels=64, 
        n_fft=512, 
        ref=ref, 
        amin=amin, 
        top_db=top_db
    )
    transform = lambda x: mel(spectrogram(x)).squeeze(1)
    
    filenames = []
    files = glob.glob(os.path.join(args.root, 'wav', '*.wav'))
    for counter, file in tqdm.tqdm(enumerate(files), total=len(files), desc='Melspects'):
        audio, fs = af.read(
            file,
            always_2d=True
        )
        if fs != 16000:
            audio = torchaudio.transforms.Resample(fs, 16000)(torch.from_numpy(audio))
        else:
            audio = torch.from_numpy(audio)
        logmel = transform(audio)
        filename = os.path.join(args.dest, '{:012}'.format(counter))
        np.save(filename, logmel)
        filenames.append(filename)
    features = pd.DataFrame(
        data=filenames,
        index=pd.Index([os.path.basename(x) for x in files], name='file'),
        columns=['features']
    )
    features.reset_index().to_csv(os.path.join(args.dest, 'features.csv'), index=False)