# Fashion-MNIST Classification with PyTorch MLP

This repository contains a PyTorch implementation of a Multilayer Perceptron (MLP) for classification on the Fashion-MNIST dataset from CSV data, featuring a modular data-processing and training pipeline.

---

## 📌 Project Overview

- **Custom `Dataset` Class (`MYCSVDataSet`):**  
  - Automatic detection and handling of numerical and categorical columns.
  - Missing value imputation via `SimpleImputer`.
  - Feature normalization using `StandardScaler` and categorical encoding via `OneHotEncoder`.
- **Modular `Trainer` Class:**  
  - Streamlined training and validation loops[cite: 2].
  - Automatic checkpoint saving (`best_model.pt`) tracking minimum validation loss (`val_loss`)[cite: 2].
- **Hyperparameter Exploration:**  
  - Systematic comparison across multiple batch sizes (16, 32, 64) and learning rates (0.01, 0.001, 0.0001)[cite: 2].

---

## 🧱 Model Architecture

Designed for 28x28 flattened grayscale images (784 features)[cite: 2]:

```text
Input (784) ──► Linear(784, 32) ──► ReLU ──► Linear(32, 10) ──► CrossEntropyLoss
