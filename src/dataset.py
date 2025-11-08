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
