from torch import nn
import torch

class VGGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vggBlock1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3),
        ) # 32 * 16 * 16
        self.vggBlock2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.4),
        ) # 64 * 8 * 8
        self.vggBlock3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.4),
        ) # 128 * 4 * 4
        self.flatten = nn.Flatten()
        self.linearBlock = nn.Sequential(
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 10),
        )
    
    def forward(self, x:torch.tensor):
        x = self.vggBlock1(x)
        x = self.vggBlock2(x)
        x = self.flatten(x)
        x = self.linearBlock(x)
        return x