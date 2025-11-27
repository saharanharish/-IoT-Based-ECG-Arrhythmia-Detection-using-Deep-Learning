
# 📘 **ECG Heartbeat Classification Using 1D CNN (MIT-BIH Dataset)**

*A Deep Learning Project for ECG Arrhythmia Detection*

---

## 🧩 **Project Overview**

This project implements a **1D Convolutional Neural Network (1D-CNN)** for classifying ECG heartbeats into **five categories** using the MIT-BIH Arrhythmia dataset (187-sample beat format).

Since only one dataset file (`mitbih_test.csv`) was available, the pipeline uses:

* Manual **train/test split**
* Standardization
* Reshaping for Conv1D
* One-hot encoding
* CNN training & evaluation
* Confusion matrix + classification report

This project is designed to be **simple, clear, and easy to understand**, making it suitable for:

* Academic submissions
* Portfolio / GitHub projects
* Interview demonstrations
* IoT healthcare integration

---

## 📂 **Repository Structure**

```
├── data/
│   └── mitbih_test.csv          # ECG dataset (187 features + label)
├── notebook/
│   └── ecg_cnn_training.ipynb   # Colab notebook with all steps
├── models/
│   └── ecg_cnn_model.h5         # Saved CNN model (optional)
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

---

# 🔹 **Dataset Description**

Each row in `mitbih_test.csv` represents:

* **187 values** → ECG signal samples of one heartbeat
* **1 label** → class (0-4)

Dataset shape example:

```
(21891, 188)
```

---

# 🧱 **Pipeline Overview**

The complete machine learning pipeline consists of:

1. Import required libraries
2. Upload & unzip dataset
3. Load CSV into DataFrame
4. Split into `X` (features) and `y` (labels)
5. Train/test split
6. Standardization
7. Reshaping for 1D-CNN
8. One-hot encoding
9. Model building
10. Training
11. Evaluation
12. Confusion matrix & classification report

All code is included below for clarity.

---

# 🧪 **Code Implementation**

### ✅ **0. Import Libraries**

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
```

---

### ✅ **1. Upload & Unzip Dataset**

```python
from google.colab import files
uploaded = files.upload()

!unzip mitbih_test.zip
```

---

### ✅ **2. Load Dataset**

```python
df = pd.read_csv("mitbih_test.csv", header=None)
print(df.shape)
df.head()
```

---

### ✅ **3. Split Into Features and Labels**

```python
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

---

### ✅ **4. Train/Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

### ✅ **5. Preprocessing**

#### **5.1 Standardization**

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### **5.2 Reshape for CNN**

```python
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], 187, 1)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0], 187, 1)
```

#### **5.3 One-Hot Encoding**

```python
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)
```

---

### ✅ **6. Build 1D CNN Model**

```python
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(187,1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.summary()
```

---

### ✅ **7. Compile & Train**

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_cnn, y_train_cat,
    validation_split=0.2,
    epochs=15,
    batch_size=128,
    verbose=1
)
```

---

### ✅ **8. Evaluate Model**

```python
test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)
```

---

### ✅ **9. Confusion Matrix & Classification Report**

```python
y_pred_prob = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_prob, axis=1)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

# 📊 **Model Performance**

Your results were:

* **Accuracy:** ~97%
* **Weighted F1-score:** ~0.97
* **Strong performance** on classes 0, 2, and 4
* **Weaker recall** on minority classes (1 & 3) due to imbalance

![ECG Output](https://github.com/saharanharish/-IoT-Based-ECG-Arrhythmia-Detection-using-Deep-Learning/blob/main/Screenshot%202025-11-27%20122206.png)


> "I trained a 1D CNN on MIT-BIH ECG heartbeat data.
> Each heartbeat consists of 187 samples.
> I standardized the dataset, reshaped it for Conv1D, one-hot encoded labels, and trained a CNN with two convolutional layers.
> The model achieved ~97% accuracy.
> I evaluated model performance using a confusion matrix and classification report to analyze class-wise metrics, especially for minority arrhythmias."

---

# 🚀 **Future Work**

* Implement class balancing
* Try deeper architectures (CNN-LSTM)
* Integrate with IoT pipeline (AD8232 → ESP32 → MQTT)

---

# 📜 **License**

This project is open-source and free to use for research and education.


