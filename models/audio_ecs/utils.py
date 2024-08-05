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
from transformers import BertModel, BertConfig

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOSS_CRITERIA = nn.CrossEntropyLoss()

class BertForSequenceLabeling(nn.Module):
    def __init__(self, num_labels=50):
        super(BertForSequenceLabeling, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        set_parameter_requires_grad(self.bert, feature_extracting=True)
        end_input_features = self.bert.pooler.dense.in_features
        self.bert.pooler.dense = nn.Linear(end_input_features,num_labels)
        self.num_labels = num_labels  # Store num_labels as an instance variable

    def forward(self, input_ids, attention_mask=None):
        # Transform the embeddings to BERT-compatible dimensions
        # Pass through BERT
        outputs = self.bert(inputs_embeds=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, max_length, hidden_size)
        
        # Prepare logits for sequence labeling
        logits = self.bert.pooler.dense(sequence_output)  # Shape: (batch_size, max_length, num_labels)
        return logits


class DataGenerator(Dataset):
    def __init__(self, path, kind='train', labels_amount=50):
        
        self.model = torchaudio.pipelines.WAV2VEC2_BASE.get_model().to(DEVICE)
        # print(self.model)
        if kind=='train':
            files = Path(path).glob('[1-3]-*')
            self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
        
        if kind=='val':
            files = Path(path).glob('4-*')
            self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
            
        if kind=='test':
            files = Path(path).glob('5-*')
            self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
            
        self.length = len(self.items)
        self.labels_amount = labels_amount
        
    def __getitem__(self, index):
        filename, label = self.items[index]
        label = torch.tensor(int(label), dtype=torch.long)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes=self.labels_amount)
        data_tensor, rate = torchaudio.load(filename)
        wav_to_vec_rate = 16000
        transform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=wav_to_vec_rate)
        resampled_data_tensor = transform(data_tensor)
        
        with torch.no_grad():
            self.model.eval()
            resampled_data_tensor = torch.Tensor(resampled_data_tensor).to(DEVICE)
            tmp, _ = self.model.extract_features(resampled_data_tensor)
            tmp = tmp[6]
            tmp = tmp.data.cpu().numpy()[0]
        
        # tmp = data_tensor[0,0:80000]
        return (tmp, label_one_hot)
    
    def __len__(self):
        return self.length


def set_parameter_requires_grad(model, feature_extracting=False):
    # approach 1
    if feature_extracting:
        # frozen model
        model.requires_grad_(False)
    else:
        # fine-tuning
        model.requires_grad_(True)
        
    # approach 2
    if feature_extracting:
        # frozen model
        for param in model.parameters():
            param.requires_grad = False
    else:
        # fine-tuning
        for param in model.parameters():
            param.requires_grad = True
    

def preparte_data_loader(mode='train', batch_size = 64, audio_folder= os.path.join(Path(os.getcwd()).parent.parent,"datasets/ESC-50-master/audio")):
    train_data = DataGenerator(audio_folder, kind=mode)
    # test_data = DataGenerator(audio_folder, kind='test')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader

if __name__ == '__main__':
    print()