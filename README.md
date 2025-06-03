# SQC-Automated-
Codebase for automating signal quality control in time-series data using ML models like RF, SVM, FFNN, and CNN. Includes data processing, training, and evaluation.

---

# Using Machine Learning to Automate Signal Quality Control for Large Time-Series Datasets

This repository contains the code used for a Master's research project focused on automating Signal Quality Control (SQC) of gait data collected from wearable sensors. The work explores the use of four machine learning models—**Random Forest (RF)**, **Support Vector Machine (SVM)**, **Feedforward Neural Network (FFNN)**, and **Convolutional Neural Network (CNN)**—to classify segments of walking data as either **acceptable** or **query** based on signal quality.

---

## Project Overview

Large-scale wearable sensor datasets used in biomedical research often contain noisy or artifact-ridden segments. Manual quality control is time-consuming and subjective. This project investigates a machine learning-based approach to automate the classification of signal quality in walking bouts, particularly using lower-back sensor data from older adults with proximal femoral fractures.

### Main Objectives:
- Build a labelled dataset using expert-annotated walking bouts.
- Extract meaningful time- and frequency-domain features from raw gait signals.
- Develop, train, and evaluate multiple machine learning models.
- Compare model performance and generalisability on internal and unseen data.

---

## Models Implemented

- **Random Forest (RF):** Trained on hand-crafted features; interpretable and handles noisy, high-dimensional data well.
- **Support Vector Machine (SVM):** Uses kernel methods to find optimal decision boundaries; good at generalising with fewer samples.
- **Feedforward Neural Network (FFNN):** Utilised tabular features with two hidden layers and ReLU activations.
- **Convolutional Neural Network (CNN):** Operated on spectrogram images; effective at capturing complex spatial-temporal signal patterns.

---

## Evaluation Metrics

Each model was evaluated using standard classification metrics:

- **Accuracy**  
- **Precision**  
- **Recall** (especially for detecting poor-quality ‘query’ signals)  
- **F1-Score**  
- **AUC-ROC**

---

## Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| RF    | 81.48%   | 64.52%    | 83.33% | 72.73%   | 0.8944 |
| FFNN  | 80.25%   | 60.00%    | 100%   | 75.00%   | 0.9013 |
| SVM   | 86.42%   | 74.07%    | 83.33% | 78.43%   | 0.9101 |
| CNN   | **98.29%** | **88.44%** | **100%** | **93.86%** | **0.9996** |

---

## 📈 ROC Curves

Below are the ROC curve comparisons for each model:
<img width="991" alt="Screenshot 2025-06-03 at 16 58 27" src="https://github.com/user-attachments/assets/5949843b-3643-4c1c-84e2-dddb0444853b" />

---
