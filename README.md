Name: Sayantika Chakraborty
Roll No.:ME22B190

# Assignment 3 - Clustering-Based Sampling for Fraud Detection

## Overview
This project implements and compares different **resampling techniques** to handle extreme class imbalance in the **Credit Card Fraud Detection dataset**. The dataset is highly imbalanced, with fraudulent transactions making up only ~0.17% of all records. Standard classifiers fail under this imbalance, so resampling strategies are applied to improve fraud detection.

The assignment covers:
- Logistic Regression on imbalanced data (baseline)
- SMOTE oversampling
- Clustering-Based Oversampling (CBO)
- Clustering-Based Undersampling (CBU)
- Evaluation of performance metrics (Precision, Recall, F1-score, ROC AUC)
- Analysis of threshold effects on performance

## Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
- **Size**: 284,807 transactions 
- **Frauds**: 492 (~0.172%) 
- **Imbalance Ratio**: ~577:1 (non-fraud : fraud)

The features `V1–V28` are PCA-transformed to protect privacy. Two original features `Time` and `Amount` are scaled before use.

## Methods Implemented
### 1. Baseline Model (Imbalanced)
- Logistic Regression trained directly on imbalanced data.
- Expected high accuracy, but low recall for fraud cases.

### 2. SMOTE Oversampling
- Synthetic Minority Oversampling Technique.
- Generates synthetic fraud samples to balance the dataset.
- **Limitation**: May create unrealistic samples in sparse or overlapping regions.

### 3. Clustering-Based Oversampling (CBO)
- Minority class (fraud) clustered using K-Means.
- Oversampling performed within each cluster proportionally.
- Preserves subgroup diversity and avoids unrealistic samples.

### 4. Clustering-Based Undersampling (CBU)
- Majority class (non-fraud) clustered using K-Means.
- Undersampling performed proportionally within each cluster.
- Reduces dataset size while preserving distribution.

## Evaluation Metrics
Since the dataset is highly imbalanced, **accuracy is misleading**. 
We use the following metrics for the **fraudulent (minority) class**:
- **Precision**: How many predicted frauds are truly fraud. 
- **Recall (Sensitivity)**: How many actual frauds are caught. 
- **F1-score**: Harmonic mean of precision and recall. 
- **ROC AUC**: Probability that the model ranks a fraud higher than a non-fraud.

## Results Summary
| Model                 | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) | ROC AUC |
|------------------------|------------------:|---------------:|-----------------:|--------:|
| Baseline (Imbalanced)  | 0.829             | 0.643          | 0.724            | 0.957   |
| SMOTE (Tuned)          | 0.833             | 0.816          | 0.825            | 0.970   |
| CBO (Tuned)            | 0.833             | 0.816          | 0.825            | 0.972   |
| CBU (Tuned)            | 0.723             | 0.827          | 0.771            | 0.972   |

## Key Insights
- **Baseline**: High precision but misses many frauds (low recall). 
- **SMOTE**: Boosts recall, but needs threshold tuning to fix precision issues. 
- **CBO**: Matches SMOTE performance, but is conceptually safer since it respects fraud cluster structure. 
- **CBU**: Improves recall, reduces dataset size, but precision is lower compared to oversampling. 

## Conclusion
- **Best Method**: Clustering-Based Oversampling (CBO). 
  - Achieves strong balance between precision and recall. 
  - Avoids SMOTE’s limitation of creating unrealistic synthetic points. 
- **Recommendation**: The company should adopt **CBO** for fraud detection. 
- **Threshold Tuning is Critical**: Without adjusting the decision threshold, resampling models show very poor precision and F1-scores.

## How to Run
1. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud

