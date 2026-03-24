import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

def extract_features(file_path, max_len=128):
    y, sr = librosa.load(file_path, sr=16000)
    
    # Pre-pad very short audio signals to prevent librosa's n_fft warning
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')
        
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048)
    
    # Normalize features to prevent the CNN from collapsing to a single prediction
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    
    # Pad or truncate to max_len
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
        
    return np.expand_dims(mfcc, axis=0) # Add channel dimension for CNN

class VoiceDataset(Dataset):
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.samples = []
        self.cache = {} # Caches loaded audio to dramatically speed up training
        
        real_dir = os.path.join(data_dir, "REAL")
        fake_dir = os.path.join(data_dir, "FAKE")
        
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(".wav"):
                    self.samples.append((os.path.join(real_dir, file), 0))
                    
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(".wav"):
                    self.samples.append((os.path.join(fake_dir, file), 1))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        if idx not in self.cache:
            mfcc = extract_features(file_path)
            self.cache[idx] = torch.tensor(mfcc, dtype=torch.float32)
            
        return self.cache[idx], torch.tensor(label, dtype=torch.long)
