# SVM Chest X-Ray Pneumonia Classification

This repository contains the implementation for two assignments in the Statistical Machine Learning course.

## Assignment 1

Implement Soft-margin SVM from scratch using NumPy and train it using SGD on the Chest X-Ray Images (Pneumonia) dataset.

## Assignment 2

Implement SVM using a machine learning library, specifically scikit-learn, and compare the result with the implemented SVM from Assignment 1.

## Dataset

The dataset is not included in this repository due to file size.

Dataset link: Chest X-Ray Images (Pneumonia) on Kaggle  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

After downloading the dataset, update `BASE_DIR` in the notebooks to match the local dataset path.

Example:

```python
BASE_DIR = r"C:\Users\Admin\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray\chest_xray"