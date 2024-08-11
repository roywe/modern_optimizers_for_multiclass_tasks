# Import PyTorch libraries
import torch
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import torchaudio
import numpy as np
import pandas as pd

DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
LOSS_CRITERIA = nn.CrossEntropyLoss()
ONLY_10_LABELS = True



class AudioDataset(Dataset):
    def __init__(self, kind='train', path=os.path.join(Path(os.getcwd()).parent.parent,"datasets/ESC-50-master/audio"), labels_amount = 50, is_only_10=ONLY_10_LABELS):
        if is_only_10:
            meta_data_file = os.path.join(Path(os.getcwd()).parent.parent,"datasets/ESC-50-master/meta",'esc50.csv')
            dataset = pd.read_csv(meta_data_file)
            valid_files = list(dataset[dataset['esc10']]['filename'])
            # labels_amount = 10
            
        self.labels_amount = labels_amount
        if kind=='train':
            files = Path(path).glob('[1-3]-*')
            self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
        
        if kind=='val':
            files = Path(path).glob('4-*')
            self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
            
        if kind=='test':
            files = Path(path).glob('5-*')
            self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
        
        if is_only_10:
            self.items = [file for file in self.items if Path(file[0]).name in valid_files]
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        filename, label = self.items[idx]
        label = torch.tensor(int(label), dtype=torch.long)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes=self.labels_amount)
        data_tensor, rate = torchaudio.load(filename)
        wav_to_vec_rate = 16000
        transform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=wav_to_vec_rate)
        data_tensor = transform(data_tensor)
        data_tensor = data_tensor.squeeze(0)
        # Convert waveform to feature representation if needed (e.g., MFCC, spectrogram)
        # Here we'll just return the raw waveform for simplicity
        return data_tensor, label_one_hot  # [sequence_length]
    # [:40000]
# [:40000]
if __name__ == '__main__':
    print()