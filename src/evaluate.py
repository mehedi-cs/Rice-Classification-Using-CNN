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

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.show()

    print(f"Average Test Loss: {np.mean(losses):.4f}")
