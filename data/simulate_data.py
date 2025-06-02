import numpy as np
import pandas as pd
import os

def simulate_driving_data(num_samples=10000):
    np.random.seed(42)
    time = np.arange(0, num_samples * 0.1, 0.1)  # 10 Hz

    def gen_pattern(label):
        speed = np.clip(np.random.normal(loc=60, scale=10, size=num_samples), 0, 150)
        accel = np.gradient(speed)
        brake = np.random.choice([0, 1], size=num_samples, p=[0.97, 0.03]) if label == 'aggressive' else np.zeros(num_samples)
        turn_rate = np.random.normal(loc=0.2 if label == 'aggressive' else 0.05, scale=0.1, size=num_samples)
        return pd.DataFrame({
            'time': time,
            'speed': speed,
            'accel': accel,
            'brake': brake,
            'turn_rate': turn_rate,
            'label': label
        })

    df = pd.concat([
        gen_pattern('aggressive'),
        gen_pattern('smooth'),
        gen_pattern('distracted')
    ]).reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/obd2_simulated.csv", index=False)
    print("Simulated data saved to data/raw/obd2_simulated.csv")

if __name__ == "__main__":
    simulate_driving_data()
