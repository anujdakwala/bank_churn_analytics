# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 17:31:15 2025

@author: adakw
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import pickle

# Load data
train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTrain data columns:", train_df.columns)
print("\nMissing values in train data:\n", train_df.isnull().sum())
print("\nMissing values in test data:\n", test_df.isnull().sum())


# Handle missing values (none in our sample, but would implement if needed)
# train_df.fillna(method='ffill', inplace=True)

# Handle special characters in surnames (like K? and Ch'ang)
train_df['Surname'] = train_df['Surname'].str.replace('[^a-zA-Z\']', '', regex=True)
test_df['Surname'] = test_df['Surname'].str.replace('[^a-zA-Z\']', '', regex=True)

# Convert categorical variables
label_encoder = LabelEncoder()
for col in ['Geography', 'Gender']:
    train_df[col] = label_encoder.fit_transform(train_df[col])
    test_df[col] = label_encoder.transform(test_df[col])
    
    
    
    
# Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=train_df)
plt.title('Churn Distribution')
plt.show()

# Numerical features distribution
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
train_df[num_cols].hist(bins=20, figsize=(15,10))
plt.suptitle('Numerical Features Distribution')
plt.show()

# Categorical features vs churn
cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.barplot(x=col, y='Exited', data=train_df)
    plt.title(f'Churn Rate by {col}')
    plt.show()

# Correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()    




# Create new features
def create_features(df):
    # Balance to salary ratio
    df['BalanceSalaryRatio'] = df['Balance']/(df['EstimatedSalary']+1)
    
    # CreditScore to Age ratio
    df['CreditScoreToAge'] = df['CreditScore']/(df['Age']+1)
    
    # Interaction terms
    df['IsActive_Gender'] = df['IsActiveMember'] * df['Gender']
    df['HasCrCard_Balance'] = df['HasCrCard'] * (df['Balance']>0).astype(int)
    
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

# Features and target
X = train_df.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = train_df['Exited']

# Test data (without Exited as it's not in the file)
X_test_final = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'BalanceSalaryRatio', 'CreditScoreToAge']
X_train_smote[num_cols] = scaler.fit_transform(X_train_smote[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test_final[num_cols] = scaler.transform(X_test_final[num_cols])



# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]
    
    results[name] = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'ROC AUC': roc_auc_score(y_val, y_prob),
        'Classification Report': classification_report(y_val, y_pred),
        'Confusion Matrix': confusion_matrix(y_val, y_pred)
    }
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {results[name]['Accuracy']:.4f}")
    print(f"ROC AUC: {results[name]['ROC AUC']:.4f}")
    print("Classification Report:\n", results[name]['Classification Report'])




import joblib
# Get best model (assuming XGBoost performed best)
best_model = models['XGBoost']

# Feature importance
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12,8))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title('Feature Importance')
plt.show()



# Make predictions on test data
test_predictions = best_model.predict(X_test_final)
test_probabilities = best_model.predict_proba(X_test_final)[:,1]

# Add predictions to test data
test_df['ChurnProbability'] = test_probabilities
test_df['PredictedChurn'] = test_predictions

# Save predictions
test_df[['id', 'CustomerId', 'Surname', 'ChurnProbability', 'PredictedChurn']].to_csv('churn_predictions.csv', index=False)

# Business insights
high_risk_customers = test_df[test_df['ChurnProbability'] > 0.7].sort_values('ChurnProbability', ascending=False)
print(f"\nNumber of high-risk customers (probability > 70%): {len(high_risk_customers)}")

# Key drivers of churn
print("\nTop factors contributing to churn:")
for feature, importance in zip(X.columns, best_model.feature_importances_):
    if importance > 0.05:  # Only show significant features
        print(f"{feature}: {importance:.2f}")

joblib.dump(best_model, r"scaler.pkl")
