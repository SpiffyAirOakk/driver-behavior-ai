import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
