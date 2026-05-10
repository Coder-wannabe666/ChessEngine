import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        # Convolutional layers (pattern recognition)
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        # Activation function
        self.relu = nn.ReLU()

        # Fully connected layers (decision making)
        # 128 channels * 8 * 8 board size = 8192 inputs
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1) # Single output (evaluation)

        # Output activation (Tanh compresses output between -1 and 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Pass input through conv layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Flatten 3D tensor to 1D vector
        x = x.view(x.size(0), -1)

        # Pass through dense layers
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))

        return x