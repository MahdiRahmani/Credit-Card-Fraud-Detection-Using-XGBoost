# Credit Card Fraud Detection Using XGBoost

## Problem Statement
Credit card fraud detection is a significant challenge, especially due to the highly imbalanced nature of fraud transactions. The goal of this project is to accurately detect fraudulent transactions to minimize the loss incurred by both customers and financial institutions. The dataset used in this project is sourced from Kaggle and contains credit card transactions made by European cardholders in September 2013.

[Dataset Reference from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Solution Overview
This project utilizes the XGBoost algorithm to build a robust model capable of detecting fraudulent transactions. The key challenge is handling the class imbalance, where fraudulent transactions make up only a very small fraction of the total data. We use SMOTE (Synthetic Minority Over-sampling Technique) to address this imbalance and perform hyperparameter tuning on XGBoost to optimize the model's performance.

## Methodology
1. **Data Preprocessing**: The dataset consists of anonymized features, `Time` and `Amount`. Feature scaling is applied to `Time` and `Amount` to normalize these features.
2. **Outlier Detection**: Isolation Forest is used to remove outliers from the non-fraudulent transactions to reduce noise.
3. **Class Imbalance Handling**: The SMOTE technique is used to oversample the minority class (fraudulent transactions) to achieve a more balanced dataset.
4. **Model Training and Hyperparameter Tuning**: The XGBoost classifier is used, and hyperparameter tuning is conducted to find the optimal settings for parameters like `n_estimators`, `max_depth`, `learning_rate`, and others.
5. **Evaluation**: The Area Under the Precision-Recall Curve (AUPRC) is used to evaluate the model's performance due to the imbalanced nature of the dataset. The precision-recall curve is plotted, and the classification report is generated to understand the model's efficacy in detecting fraud.

## Libraries Used
- **Numpy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting graphs and visualizations.
- **Scikit-learn**: For data preprocessing, metrics, and sampling techniques (e.g., SMOTE, IsolationForest).
- **Imbalanced-learn**: For handling class imbalance using SMOTE.
- **XGBoost**: To train a highly efficient gradient boosting model for fraud detection.

## Results
The model was tuned using a grid search over important hyperparameters. The best model achieved an AUPRC score of `X.XXXX`. Below are key metrics of the model:
- **Best Parameters**: Hyperparameters such as `n_estimators`, `max_depth`, `learning_rate`, etc., were tuned to optimize the performance.
- **Precision-Recall Curve**: The precision-recall curve was plotted for the best model to visualize the performance in identifying fraudulent transactions.
- **Classification Report**: The classification report includes precision, recall, and F1-score for both the majority and minority classes, showing how well the model performed in fraud detection.

For more detailed results and the code, please refer to the repository files.

## Reference
- Dataset used for this project: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

