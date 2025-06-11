# EMPLOYEE TURNOVER PREDICTION WITH XAI 

## Project Overview

This project, supported by TÜBİTAK, aims to predict employee turnover using machine learning techniques and evaluate the effects of preprocessing, class balancing, discretization, and explainability on model performance and transparency.

## Project Contents 🌟

![]([https://github.com/aysenurkocaak/photo](https://github.com/aysenurkocaak/photo/blob/main/Employee_turnover_prediction%20(4)-1.png))


## Workflow Summary:
## 1. Data Preprocessing:
The IBM HR Analytics dataset was cleaned and preprocessed to prepare for modeling. Missing values and class imbalance were analyzed and addressed.
## 2. Modeling:
Several machine learning models were trained and evaluated. Random Forest achieved the best performance and was selected as the primary model for subsequent stages.

## 3.SMOTE Techniques for Class Balancing:
Multiple SMOTE-based oversampling techniques were tested to handle class imbalance:

SMOTE-NC achieved the highest F1-score (0.93) among all techniques.

However, SMOTE-NC requires the entire dataset to be fully categorical. Since our dataset did not meet this condition, it was not suitable for practical implementation.

Therefore, we proceeded with KMeans-SMOTE, the next best-performing method that fit our dataset structure.

## 4. Discretization Methods (Tested Separately):
We evaluated seven different discretization techniques independently, without combining them with SMOTE, to assess their individual effects:

### Equal Width Binning

### Equal Frequency Binning

### KMeans Discretization

### KMeans++ (Fast Init)

### KMeans (Elbow Method)

### Decision Tree Discretization

### Normalized Decision Tree Discretization

➤ The KMeans++ method achieved the best result, increasing the F1-score up to 84.6%, demonstrating the importance of feature transformation.

## 5. Explainable AI (XAI):
To make model predictions more interpretable and transparent for human resource decision-makers, we applied:

### LIME
for local interpretability (individual predictions)

### SHAP
for both global and local feature importance analysis


