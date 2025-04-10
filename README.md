# Pneumonia Classification from Chest X-Rays

![Confusion Matrix](results/confusion_matrix.png)

## 📌 Overview
PyTorch-based CNN model to classify Chest X-Ray images into `NORMAL` or `PNEUMONIA` using transfer learning (ResNet18).

## 🛠️ Installation

git clone https://github.com/yourusername/pneumonia-classification.git
cd pneumonia-classification
pip install -r requirements.txt

## 📂 Dataset
Download from Kaggle and place in data/ folder:

chest_xray/
  ├── train/
  ├── test/
  └── val/

## 🚀 Training

 Script/pneumonia_classification.py

## 📊 Results
Metric	    Value
Accuracy	 92%
Precision	 0.96
Recall	     0.93

## 🎛️ Hyperparameters
Batch Size: 32
Epochs: 10
Optimizer: Adam (LR=0.001)
Loss: Binary Cross-Entropy

## 🤖 Model Architecture

ResNet18(
  (fc): Sequential(
    (0): Linear(in_features=512, out_features=256)
    (1): ReLU()
    (2): Dropout(p=0.2)
    (3): Linear(in_features=256, out_features=1)
    (4): Sigmoid()
  )
)

## 📜 License
MIT
---
### **Key GitHub Additions**
1. **Visuals**: Add plots (confusion matrix, loss curves) in `results/`.
2. **Badges**: Add shields.io badges for PyTorch version, license, etc.
   ```markdown
   ![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-red)