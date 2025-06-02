import torch
import torch.nn as nn

class CNN1DClassifier(nn.Module):
    def __init__(self, input_channels=4, seq_len=50, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * ((seq_len - 2)//2), num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
