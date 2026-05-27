# Credit Card Fraud Detection using Ensemble Methods & Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📌 Project Overview
This repository contains a comprehensive Machine Learning and Deep Learning pipeline designed to detect fraudulent credit card transactions. Utilizing the highly imbalanced European cardholders dataset (ULB), this project focuses on evaluating and optimizing models based on **Precision-Recall curves** and **Average Precision (AP)** rather than misleading standard accuracy.

### The Challenge: Extreme Class Imbalance
* Total Transactions: **284,807**
* Fraudulent Transactions: **492** (only **0.172%** of the total dataset)
* Since a naive baseline model guessing "never fraud" yields 99.828% accuracy, this project prioritizes maximizing **Recall** (catching frauds) while maintaining acceptable **Precision** (minimizing false alarms for customers).

---

## 🛠️ Methodology & Tech Stack
* **Data Preprocessing**: `StandardScaler` applied to *Time* and *Amount* features. Stratified 80/20 train/test split to maintain class proportions.
* **Handling Imbalance**: Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) for supervised models.
* **Unsupervised Anomaly Detection**: Evaluated an Autoencoder trained *strictly* on genuine transactions.

**Tech Stack:** Python, Scikit-Learn, XGBoost, Keras/TensorFlow, NumPy, Matplotlib.
