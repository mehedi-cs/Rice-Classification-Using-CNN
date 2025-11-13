import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2, 2), nn.BatchNorm2d(128)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 26 * 26, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
