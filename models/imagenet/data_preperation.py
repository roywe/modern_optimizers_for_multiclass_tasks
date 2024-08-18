import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split


# dataset downloaded from https://www.kaggle.com/datasets/ponrajsubramaniian/sportclassificationdataset

file_path = '/home-sipl/prj7565/Deep_Learning_prj_Tomer/sports'


def data_pre_processing(file_path, valid_split=0.15, test_split=0.15, input_size=(224, 224), image_color='rgb', batch_size=32, shuffle=True):
    # Define the color mode transformation
    if image_color == 'rgb':
        color_mode = transforms.ToTensor()
    else:
        color_mode = transforms.Grayscale()

    # Create transformations for the training dataset
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(40),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(input_size),
        transforms.ColorJitter(brightness=[0.2, 1.0]),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.3, 0.2), fill=0)]),
        color_mode,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization for pre-trained models
    ])

    # Create transformations for the validation/test dataset
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        color_mode,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization for pre-trained models
    ])

    # Load the full dataset without any transformations
    full_dataset = ImageFolder(root=file_path, transform=None)

    # Calculate the number of samples for training and validation
    valid_size = int(valid_split * len(full_dataset))
    test_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - valid_size - test_size


    # Split the dataset into training and validation sets
    train_data, val_data , test_data = random_split(full_dataset, [train_size, valid_size , test_size])

    # Apply the respective transforms
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = val_transform
    test_data.dataset.transform = val_transform

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle , num_workers=4 , pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True , num_workers=4 , pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False , num_workers=4 , pin_memory=True)
    full_loader = DataLoader(full_dataset , batch_size=batch_size , shuffle=False , num_workers=4 , pin_memory=True)

    return train_loader, val_loader , test_loader , full_loader


train_loader, val_loader , test_loader , full_dataset = data_pre_processing(file_path)


def show_images_with_labels(data_loader, num_images=9):
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))
    
    # Make sure we have at least `num_images` in the batch
    if len(images) < num_images:
        raise ValueError("The batch contains fewer images than requested.")
    
    # Convert images to numpy arrays
    images = images[:num_images]
    labels = labels[:num_images]

    # Get class names
    class_names = data_loader.dataset.classes

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    # Display images and labels
    for i in range(num_images):
        img = images[i]
        label = labels[i]
        ax = axes[i]
        
        # Convert from (C, H, W) to (H, W, C)
        img = img.permute(1, 2, 0).numpy()

        # Normalize the image to the range [0, 1]
        if img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min())
        
        # Handle grayscale images
        if img.shape[2] == 1:
            img = img.squeeze(2)  # Remove single channel
        
        # Display the image
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        
        # Set the title with class name
        ax.set_title(class_names[label.item()])
        
        # Remove axes for clarity
        ax.axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

show_images_with_labels(full_dataset , 9)

def plot_label_distribution_pie(data_loader):
    # Get all labels from the data_loader
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())
    
    # Count the occurrences of each label
    values_counter = Counter(all_labels)
    
    # Get class names
    class_names = data_loader.dataset.classes

    # Create a pie chart
    plt.figure(figsize=(10, 10))
    plt.pie([values_counter[i] for i in range(len(class_names))], labels=class_names, autopct='%1.1f%%')
    
    # Add title
    plt.title('Distribution of Pictures Across Classes')
    
    # Show the plot
    plt.show()

plot_label_distribution_pie(full_dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_finetuned_resnet18(num_classes=22):
    
    weigths = 'DEFAULT'
    # Load the pre-trained ResNet18 model
    resnet18 = models.resnet18(weights=weigths)

    
    # Modify the last fully connected layer for the desired number of classes
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

         # Freeze all layers initially
    for param in resnet18.parameters():
        param.requires_grad = False


    for param in resnet18.layer4.parameters():
        param.requires_grad = True

    for param in resnet18.fc.parameters():
        param.requires_grad = True

    return resnet18

model = create_finetuned_resnet18()

# how many weights (trainable parameters) we have in our model? 
# num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad]) 
# print("num trainable weights: ", num_trainable_params)