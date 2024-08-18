import torch
import torch.nn as nn
import torch.optim as optim
from madgrad import MADGRAD
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import optuna
from data_preperation import create_finetuned_resnet18 , train_loader , val_loader
from adan_pytorch import Adan
import pandas as pd


batch_size = 32
num_labels = 22
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_criteria = nn.CrossEntropyLoss()
NTRAILS = 100

dataloaders = {
    'train':train_loader,
    'val':val_loader
}

def optune_optimizer_for_model(trial, optimizer_name):
    
    model = create_finetuned_resnet18().to(device)

    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)  # log=True, will use log scale to interplolate between lr
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        sgd_momentum = trial.suggest_float("sgd_momentum", 8e-1, 0.9999)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=sgd_momentum, nesterov=True)
    elif optimizer_name == 'Adan':
        beta1 = trial.suggest_float("beta1", 1e-3, 1e-1)
        beta2 = trial.suggest_float("beta2", 1e-3, 1e-1)
        beta3 = trial.suggest_float("beta3", 1e-3, 1e-1)
        optimizer = Adan(model.parameters(),lr = lr,
            betas = (beta1, beta2, beta3), 
            weight_decay = 0.02         # weight decay 0.02 is optimal per author
        )
    elif optimizer_name == 'Madgrad':
        madgrad_momentum = trial.suggest_float("madgrad_momentum", 8e-1, 0.9999)
        optimizer = MADGRAD(params=model.parameters(), lr=lr, momentum=madgrad_momentum, weight_decay=0, eps= 1e-06, decouple_decay=False)
        
 
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        
        model.train()
        
        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                output = model(inputs)
                loss = loss_criteria(output, labels)
                
                # backward + optimize only if in training phase
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model.eval()
        running_corrects = 0.0
        for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    
                    #Get model outputs and calculate loss
                    output = model(inputs)

                    loss = loss_criteria(output, labels)

                    _, preds = torch.max(output, 1)
                    
                # statistics
                running_corrects += torch.sum(preds == labels)

        epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(epoch_acc, epoch)  

        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return epoch_acc
# now we can run the experiment
sampler = optuna.samplers.TPESampler()
configurations = [
                {'optimizer_name':'Adam',
                   'study_name':'adam_study'},
                  {'optimizer_name':'SGD',
                   'study_name':'sgd_study'},
                  {'optimizer_name':'Adan',
                   'study_name':'adan_study'},
                  {'optimizer_name':'Madgrad',
                   'study_name':'madgrad_study'},
                  ]
studies = {}
for config in configurations:
    
    study = optuna.create_study(study_name=config['study_name'], direction="maximize", sampler=sampler)
    study.optimize(lambda trial: optune_optimizer_for_model(trial, config['optimizer_name']), n_trials=NTRAILS, timeout=3*60*60) #trial.report
    
    study_csv = f'/home-sipl/prj7565/Deep_Learning_prj_Tomer/optuna_results/{config["study_name"]}.csv'
    study.trials_dataframe().to_csv(study_csv)
    studies[config["study_name"]] = study_csv
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")

    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))