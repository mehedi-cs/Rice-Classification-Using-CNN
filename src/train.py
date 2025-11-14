import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        t0 = datetime.now()
        train_loss, val_loss = [], []
        correct_train, total_train = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            _, pred = torch.max(preds, 1)
            correct_train += (pred == labels).sum().item()
            total_train += labels.size(0)

        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                loss = criterion(preds, labels)
                val_loss.append(loss.item())
                _, pred = torch.max(preds, 1)
                correct_val += (pred == labels).sum().item()
                total_val += labels.size(0)

        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        train_losses.append(np.mean(train_loss))
        val_losses.append(np.mean(val_loss))
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Duration: {datetime.now()-t0}")

    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.legend(); plt.title("Accuracy"); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend(); plt.title("Loss"); plt.grid(True)
    plt.savefig('results/accuracy_loss_plot.png')
    plt.show()

    return model
