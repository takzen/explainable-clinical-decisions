# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Testable Functions (Copied from model_trainer.py) ---
# We copy the core logic into functions here to test it in isolation
# without modifying the original script.

def preprocess_data_logic(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    This function CONTAINS THE COPIED LOGIC from model_trainer.py.
    """
    # --- Start of copied block ---
    df['target'] = np.where(df['num'] > 0, 1, 0)
    df = df.drop(['num', 'id', 'dataset'], axis=1)

    X = df.drop('target', axis=1)
    y = df['target']

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    X = X.astype(float)
    # --- End of copied block ---
    
    return X, y

# --- Pytest Fixture and Tests ---

@pytest.fixture
def raw_dataframe() -> pd.DataFrame:
    """
    Provides a sample raw DataFrame for testing, mimicking the original CSV.
    """
    data = {
        'id': [1, 2, 3, 4, 5],
        'dataset': ['A', 'A', 'B', 'B', 'C'],
        'age': [63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 0],
        'cp': [3, 0, 0, 2, 1],
        'trestbps': [145, 160, 120, 130, 130],
        'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 0, 0, 1, 0],
        'thalch': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
        'slope': [0, 1, 1, 0, 2],
        'ca': [0, 3, 2, 0, 0],
        'thal': ['1', '2', '2', '2', '?'], # Mix of string and missing
        'num': [1, 2, 3, 0, 0] # Original multi-class target
    }
    return pd.DataFrame(data)

def test_copied_preprocess_data_logic(raw_dataframe):
    """
    Tests the COPIED preprocessing logic to ensure it works as expected.
    """
    # We test our new function that contains the copied logic
    X, y = preprocess_data_logic(raw_dataframe)

    # 1. Test output types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    # 2. Test shapes
    assert X.shape[0] == 5 # 5 rows
    assert X.shape[1] == 13 # 13 feature columns
    assert y.shape[0] == 5

    # 3. Test binary target creation
    expected_target = pd.Series([1, 1, 1, 0, 0], name='target')
    assert y.equals(expected_target)

    # 4. Test if dropped columns are gone
    assert 'num' not in X.columns
    assert 'id' not in X.columns
    assert 'dataset' not in X.columns
    assert 'target' not in X.columns

    # 5. Test missing value imputation
    # The 'thal' column had a '?' which should be filled with the median
    assert not X.isnull().sum().any() # No NaNs should remain

    # 6. Test data types
    assert all(X.dtypes == float)

def test_model_can_be_trained_on_processed_data(raw_dataframe):
    """
    A simple integration test to ensure the output of our processing
    can be used to train a model without errors.
    """
    X, y = preprocess_data_logic(raw_dataframe)
    
    try:
        model = RandomForestClassifier()
        model.fit(X, y)
    except Exception as e:
        pytest.fail(f"Model training failed on preprocessed data with error: {e}")