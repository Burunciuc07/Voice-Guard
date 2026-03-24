import torch
import torch.nn as nn

class VoiceGuardCNN(nn.Module):
    def __init__(self):
        super(VoiceGuardCNN, self).__init__()
        # Input shape: (Batch, 1, 40, 128)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # Adaptive pooling ensures consistent output size regardless of input spatial dimensions
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(32, 2) # Binary classification: 0=real, 1=fake
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
