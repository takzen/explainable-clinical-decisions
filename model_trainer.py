# model_trainer.py

import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

print("--- Starting Model and Explainer Training ---")

# --- 1. Load and Prepare Data ---
try:
    data_path = 'data/heart_disease_uci.csv'
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"ERROR: Data file not found at path: {data_path}")
    print("Please make sure 'heart_disease_uci.csv' is in the 'data' folder.")
    exit() # Terminate the script if data is missing

# Convert the multi-class 'num' column into a binary 'target' (0 = no disease, 1 = disease present)
df['target'] = np.where(df['num'] > 0, 1, 0)
df = df.drop(['num', 'id', 'dataset'], axis=1)

X = df.drop('target', axis=1)
y = df['target']

# Handle missing values and perform type conversion.
# This loop replaces '?' with NaN, then converts columns to numeric and fills missing values with the median.
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce') # 'coerce' will turn anything that is not a number into NaN
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

# Final check to ensure all data is of float type
X = X.astype(float)

# Split the data
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data loaded and prepared successfully.")

# --- 2. Train the Machine Learning Model ---
print("Training RandomForestClassifier model...")
# Using proven parameters that work well for this dataset
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=5)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 3. Create and Save the SHAP Explainer ---
print("Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
print("SHAP explainer created.")

# --- 4. Save the Model and the Explainer ---
dump(model, 'trained_model.joblib')
dump(explainer, 'shap_explainer.joblib')

print("--- Model and explainer have been saved successfully. ---")
