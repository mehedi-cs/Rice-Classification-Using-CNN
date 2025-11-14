import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, criterion, classes, device):
    model.eval()
    y_true, y_pred, losses = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            _, pred = torch.max(preds, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes))
    print(f"Average Test Loss: {np.mean(losses):.4f}")
