# 🔋 ANN Regression – Power Plant Energy Prediction (PyTorch)

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN)** using **PyTorch** to predict electrical energy output (PE) of a Combined Cycle Power Plant based on environmental variables.

The objective is to understand and implement a complete Deep Learning regression pipeline including preprocessing, model building, training, validation, evaluation, and model checkpointing.

---

## 📊 Dataset Description

The dataset contains 4 environmental features:

| Feature | Description |
|----------|-------------|
| AT | Ambient Temperature |
| V | Exhaust Vacuum |
| AP | Ambient Pressure |
| RH | Relative Humidity |

**Target Variable:**
- PE – Electrical Energy Output

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Loaded dataset using Pandas
- Checked for missing values
- Split into train & test sets (80/20)
- Applied feature scaling using `StandardScaler`

### 2️⃣ Tensor Conversion
- Converted NumPy arrays to PyTorch tensors
- Reshaped target using `.view(-1,1)`
- Used `TensorDataset` and `DataLoader` for mini-batch training

### 3️⃣ Model Architecture

```text
Input Layer (4 features)
    ↓
Linear(4 → 6)
ReLU
    ↓
Linear(6 → 6)
ReLU
    ↓
Linear(6 → 1)
Output (Predicted Energy)
