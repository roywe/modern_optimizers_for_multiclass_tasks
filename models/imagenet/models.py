import torch
import torch.nn as nn
import torch.optim as optim
from adan_pytorch import Adan
import numpy as np
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from madgrad import MADGRAD
from data_preperation import train_loader , test_loader , create_finetuned_resnet18
import pickle
import schedulefree
from pathlib import Path
import os


seed = 20
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_criteria = nn.CrossEntropyLoss()
dataloaders = {
    'train':train_loader,
    'val':test_loader
}

def train_model(model, dataloaders, optimizer, num_epochs=100, device=device, loss_criteria = loss_criteria):
    since = time.time()

    val_acc_history = []
    training_acc = []
    training_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                if type(optimizer) == schedulefree.adamw_schedulefree.AdamWScheduleFree:
                    optimizer.train()
            else:
                model.eval()   # Set model to evaluate mode
                if type(optimizer) == schedulefree.adamw_schedulefree.AdamWScheduleFree:
                    optimizer.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # outputs = model(inputs)
                    output = model(inputs)
                    loss = loss_criteria(output, labels)

                    _, preds = torch.max(output, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                running_corrects += torch.sum(preds == labels)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss.append(epoch_loss)
            if phase == 'train':
                training_acc.append(epoch_acc)
                training_loss.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, training_acc, val_acc_history,training_loss,val_loss

def main():

    results_metric = {
        'Adam':{
            
        },
        'SGD':{
            
        },
        'Adan':{
            
        },
        'Madgrad':{
            
        },
        'Schedulefree':{
            
        }
    }


    optimizers = ['Adan','SGD','Adam','Madgrad', 'Schedulefree']

    for optimizer_name in optimizers:

        model = create_finetuned_resnet18().to(device) 

        final_optimizer_configurations = {
        'Adam': optim.Adam(model.parameters(), lr=0.0014), #7 min
        'SGD': torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.99898, nesterov=True), #11 min
        'Adan': Adan(model.parameters() , lr=0.0342 , betas= (0.027,0.0178,0.0522) , weight_decay=0.02), #9 min
        'Madgrad': MADGRAD(params=model.parameters(), lr=0.0043, momentum=0.9420, weight_decay=0, eps= 1e-06, decouple_decay=False),  #9 min
        'Schedulefree': schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3)
        }
        
        
        model, training_acc, val_acc_history,training_loss,val_loss = train_model(model, dataloaders,
                                    final_optimizer_configurations[optimizer_name], num_epochs=100, device=device, loss_criteria = loss_criteria)
        results_metric[optimizer_name]['training_acc'] = training_acc
        results_metric[optimizer_name]['val_acc_history'] = val_acc_history
        results_metric[optimizer_name]['training_loss'] = training_loss
        results_metric[optimizer_name]['val_loss'] = val_loss
        results_metric[optimizer_name]['model'] = model
        
        path = os.path.join(Path(os.getcwd()),"#location to save dataset") #"046211_modern_optimizers_for_pretrained_models/models/imagenet/trail_one/final_results.pickle"

        with open(path, 'wb') as handle:
            pickle.dump(results_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_loss_curve(results_metric, only_val=True):
    plt.figure(figsize=(10, 5))
    for opt in results_metric:
        d = results_metric[opt]
    
        if not only_val:
    
            training_loss = [i for i in d['training_loss']]
            plt.plot(training_loss, label=f'Train Loss {opt}')
    
        val_acc_history = [i for i in d['val_loss']]

        plt.plot(val_acc_history, label=f'Validation Loss {opt}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_accuracy_curve(results_metric, only_val=True):
    plt.figure(figsize=(10, 5))
    for opt in results_metric:
        d = results_metric[opt]
        if not only_val:
            training_loss = [i.cpu().numpy().tolist() for i in d['training_acc']]
            plt.plot(training_loss, label=f'Train accuracy {opt}')
        val_acc_history = [i.cpu().numpy().tolist() for i in d['val_acc_history']]
                    
        plt.plot(val_acc_history, label=f'Validation accuracy {opt}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def show_results():
    path = os.path.join(Path(os.getcwd()),"#location to save dataset") #"046211_modern_optimizers_for_pretrained_models/models/imagenet/trail_one/final_results.pickle"
    with open(path, 'rb') as handle:
        final_results = pickle.load(handle)
    plot_loss_curve(final_results)
    plot_accuracy_curve(final_results)

if __name__ == '__main__':
    # main()
    print()
