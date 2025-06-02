import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(file_path, window_size=50):
    df = pd.read_csv(file_path)
    features = ['speed', 'accel', 'brake', 'turn_rate']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    X, y = [], []
    label_map = {'aggressive': 0, 'smooth': 1, 'distracted': 2}

    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i:i+window_size]
        label = window['label'].mode()[0]
        X.append(window[features].values)
        y.append(label_map[label])

    return train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)
