# Fraud-Detection
Predictive Model for identifying fraudulent transactions within a financial dataset.

## 1. Data Ingestion and Preprocessing
Loaded the dataset from Google Drive into a Pandas DataFrame in Google Colab.
Checked for missing values and duplicates – found none in the dataset.
Converted data types for memory efficiency – changed 64-bit floats/integers to 32-bit and converted the transaction type column to a categorical datatype to optimize processing.
Class imbalance analysis – found a significant class imbalance, with fraudulent transactions making up only 0.13% of total transactions.
## 2. Exploratory Data Analysis (EDA)
Performed statistical analysis using .describe(), .value_counts(), and correlation matrices to understand the distribution of variables.
Visualized correlations between transaction amount, balances, and fraud occurrence using Seaborn heatmaps.
Identified key fraud patterns – observed that fraudulent transactions mainly occurred in CASH_OUT and TRANSFER transactions.
## 3. Feature Engineering
Transformed categorical variables using One-Hot Encoding (type, nameOrig, nameDest).
Applied regex transformations to standardize nameOrig and nameDest, replacing customer IDs with ‘C’ and merchant IDs with ‘M’ to retain information without exposing unnecessary details.
Addressed skewness – applied log transformations and Box-Cox transformations to highly skewed numerical features (amount, oldbalanceOrg, newbalanceOrig, etc.).
Dropped redundant columns that were highly correlated (e.g., oldbalanceOrg and newbalanceOrig).
## 4. Handling Class Imbalance
Since fraudulent transactions were highly underrepresented (0.13%), standard models would perform poorly.
Applied SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic fraudulent transactions.
Combined SMOTE with Random Under-Sampling – reduced the majority class (non-fraudulent transactions) while ensuring sufficient fraud samples.
## 5. Model Training & Evaluation
Split dataset into an 80-20 train-test split.
Trained multiple models:
Decision Tree – Achieved 99.4% accuracy, but high recall was needed.
Random Forest – Improved recall but still had class imbalance issues.
Logistic Regression – Poor fraud detection due to linear limitations.
XGBoost – Performed best, achieving 99.7% accuracy with 99.4% recall, meaning most fraudulent cases were correctly identified.
Used K-Fold Cross-Validation to ensure stability and reliability of model results.
Evaluated models using multiple metrics: Precision, Recall, F1-score, ROC curve. Prioritized recall over accuracy to minimize false negatives (missed fraud cases).
## 6. Model Deployment & Interpretation
Generated feature importance scores from XGBoost to understand which features contributed most to fraud detection.
Saved and exported the final trained model for potential real-world deployment.
Plotted confusion matrices & ROC curves to visualize model performance.
## Final Outcome
Successfully built a high-performance fraud detection system.
Balanced class distribution and improved fraud detection using SMOTE & under-sampling.
XGBoost provided the best results with 99.7% accuracy and 99.4% recall.
