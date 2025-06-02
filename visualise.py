import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from utils.preprocess import load_and_preprocess
from models.lstm_model import LSTMClassifier
from models.cnn1d_model import CNN1DClassifier
import argparse
import numpy as np

LABELS = ["aggressive", "smooth", "distracted"]

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y_true, y_pred):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.countplot(x=LABELS, order=LABELS, ax=axs[0])
    axs[0].set_title("True Labels")

    pred_counts = [list(y_pred).count(i) for i in range(3)]
    sns.barplot(x=LABELS, y=pred_counts, ax=axs[1])
    axs[1].set_title("Predicted Labels")

    plt.tight_layout()
    plt.show()

def main(model_type='lstm'):
    X_train, X_test, y_train, y_test = load_and_preprocess("data/raw/obd2_simulated.csv")

    if model_type == 'lstm':
        model = LSTMClassifier()
    else:
        model = CNN1DClassifier()

    model.load_state_dict(torch.load(f"model_{model_type}.pt"))
    model.eval()

    with torch.no_grad():
        y_pred = model(torch.Tensor(X_test)).argmax(dim=1).numpy()

    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    plot_confusion_matrix(y_test, y_pred, model_type.upper())
    plot_class_distribution(y_test, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lstm', 'cnn'], default='lstm')
    args = parser.parse_args()
    main(args.model)
