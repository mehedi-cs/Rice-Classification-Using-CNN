import os
import random
import splitfolders
import torchvision
import torchvision.transforms as transforms
import torch
from termcolor import colored

def split_dataset(base_dir, output_dir='Rice_images'):
    splitfolders.ratio(input=base_dir, output=output_dir, seed=42, ratio=(0.7, 0.15, 0.15))
    print(colored("Dataset split into train/val/test", 'green', attrs=['bold']))
    return output_dir

def create_data_loader(dataset_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.class_to_idx
