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
#if we want to train for 50 classes put False here
ONLY_10_LABELS = True

class AudioDataset(Dataset):
    def __init__(self, kind='train', path=os.path.join(Path(os.getcwd()).parent.parent,"datasets/ESC-50-master/audio"), labels_amount = 50, is_only_10=ONLY_10_LABELS):
        self.labels_amount = labels_amount
        
        # if only 10 will transform audio and labels to only 10 labels instead of 50, it was neseccary to handle our compute resources
        if is_only_10:
            meta_data_file = os.path.join(Path(os.getcwd()).parent.parent,"datasets/ESC-50-master/meta",'esc50.csv')
            dataset = pd.read_csv(meta_data_file)
            valid_files = list(dataset[dataset['esc10']]['filename'])
            self.labels_amount = 10
            
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
        
        self.audio_tensors = []
        self.labels = []
        different_labels = set({})
        for filename, label in self.items:
            different_labels.add(int(label))
        
        if is_only_10:
            different_labels = sorted(different_labels)
            self.map_classes = {j:i for i, j in enumerate(different_labels)}
        
        for filename, label in self.items:
            if is_only_10:
                label = self.map_classes[int(label)]
            label = torch.tensor(int(label), dtype=torch.long)
            label_one_hot = torch.nn.functional.one_hot(label, num_classes=self.labels_amount)
            # to use hubert 16000 hz is needed threfore we need to resample audio
            data_tensor, rate = torchaudio.load(filename)
            wav_to_vec_rate = 16000
            transform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=wav_to_vec_rate)
            data_tensor = transform(data_tensor)
            data_tensor = data_tensor.squeeze(0)
            self.audio_tensors.append(data_tensor)
            self.labels.append(label_one_hot)
            
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        #take first audio - 40000 from 80000 to match compute resources,
        # we can see in the data preperation that audio distribute for all frames so taking half will not change performance dramtically
        data_tensor, label_one_hot = self.audio_tensors[idx],self.labels[idx]
        return data_tensor[:40000], label_one_hot 
# [:40000]
if __name__ == '__main__':
    print()