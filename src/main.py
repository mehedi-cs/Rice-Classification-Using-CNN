import torch
from src.utils import setup_environment
from src.dataset import split_dataset, create_data_loader
from src.model import CNN
from src.train import train_model
from src.evaluate import evaluate_model
import torch.nn as nn
import torch.optim as optim

setup_environment()

base_dir = 'data/Rice_Image_Dataset'
out_dir = split_dataset(base_dir)

train_loader, _ = create_data_loader(f'{out_dir}/train')
val_loader, _ = create_data_loader(f'{out_dir}/val')
test_loader, class_idx = create_data_loader(f'{out_dir}/test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(class_idx)
model = CNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

evaluate_model(model, test_loader, criterion, list(class_idx.keys()), device)
