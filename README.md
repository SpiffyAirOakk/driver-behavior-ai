
# 🚗 Driver Behavior Classifier (from OBD2-style Data)

Classify driving behavior into **aggressive**, **smooth**, or **distracted** using AI on time-series telematics data.


---

## 📌 Project Overview

This project demonstrates how machine learning can be applied to **OBD2-style vehicle data** to classify different types of driver behavior. It simulates real-world scenarios using time-series data (speed, acceleration, braking, and turn rate) and applies deep learning models to make behavior predictions.

### 🧠 Why This Matters

Understanding driver behavior is critical for:
- **Usage-based insurance**
- **Fleet safety monitoring**
- **Driver coaching and assistance systems**
- **Smart vehicle telematics**

---

## 🛠️ Features

- ✅ Simulated & real-like OBD2 data
- ✅ LSTM and 1D CNN models for time-series classification
- ✅ Evaluation metrics (precision, recall, F1-score)
- ✅ Visualization tools: confusion matrix & class distributions
- ✅ Clean modular structure with training, evaluation, and visualization scripts

---

## 📁 Folder Structure

```
driver-behavior-ai/
├── data/
│   └── raw/obd2_simulated.csv         # Simulated driving data
├── models/
│   ├── lstm_model.py                  # LSTM implementation
│   └── cnn1d_model.py                 # 1D CNN implementation
├── utils/
│   └── preprocess.py                  # Data loader & preprocessor
├── train.py                           # Model training
├── evaluate.py                        # Evaluation metrics
├── visualize.py                       # Plots & metrics visualizations
├── requirements.txt                   # Dependencies
└── README.md
```

---

## 📊 Example Results

```
              precision    recall  f1-score   support

  aggressive       1.00      1.00      1.00        43
      smooth       0.44      0.56      0.49        36
  distracted       0.50      0.39      0.44        41

    accuracy                           0.66       120
   macro avg       0.65      0.65      0.64       120
weighted avg       0.66      0.66      0.66       120
```

🎯 Model is excellent at classifying **aggressive** behavior. Further tuning is needed to better distinguish between **smooth** and **distracted** drivers.

---

## 🔧 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/driver-behavior-ai.git
cd driver-behavior-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
# LSTM model
python train.py --model lstm

# Or 1D CNN model
python train.py --model cnn
```

### 4. Evaluate Performance

```bash
python evaluate.py --model lstm
```

### 5. Visualize Results

```bash
python visualize.py --model lstm
```

---

## 🔍 Data Description

Each record is a time-series window with the following features:

| Feature        | Description                   |
|----------------|-------------------------------|
| `speed`        | Vehicle speed (km/h)          |
| `acceleration` | Longitudinal acceleration     |
| `brake`        | Braking force (0-1 scale)     |
| `turn_rate`    | Angular velocity (degrees/sec)|
| `label`        | Driver behavior class         |

---

## 📈 Model Architectures

### 📌 LSTM
- Ideal for sequential data
- Learns temporal dependencies

### 📌 CNN 1D
- Fast and effective for time windows
- Captures local driving patterns

---

## 📌 Future Improvements

- Add **GPS jerk** and **steering angle** features
- Include real-world driving datasets (e.g., UAH-DriveSet)
- Hyperparameter tuning & class balancing
- Streamlit dashboard for live monitoring

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new ideas.

---

## 📄 License

MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Made with ❤️ by [Abdullah Khalid](https://github.com/SpiffyAirOakk)  

