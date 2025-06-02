import torch
from sklearn.metrics import classification_report
from utils.preprocess import load_and_preprocess
from models.lstm_model import LSTMClassifier
from models.cnn1d_model import CNN1DClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['lstm', 'cnn'], default='lstm')
args = parser.parse_args()

X_train, X_test, y_train, y_test = load_and_preprocess("data/raw/obd2_simulated.csv")

model = LSTMClassifier() if args.model == 'lstm' else CNN1DClassifier()
model.load_state_dict(torch.load(f"model_{args.model}.pt"))
model.eval()

with torch.no_grad():
    preds = model(torch.Tensor(X_test)).argmax(dim=1).numpy()

print(classification_report(y_test, preds, target_names=["aggressive", "smooth", "distracted"]))
