import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils.preprocess import load_and_preprocess
from models.lstm_model import LSTMClassifier
from models.cnn1d_model import CNN1DClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['lstm', 'cnn'], default='lstm')
args = parser.parse_args()

X_train, X_test, y_train, y_test = load_and_preprocess("data/raw/obd2_simulated.csv")

train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = LSTMClassifier() if args.model == 'lstm' else CNN1DClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), f"model_{args.model}.pt")
