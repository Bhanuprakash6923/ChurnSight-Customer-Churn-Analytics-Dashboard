import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,   # 👈 added
)
import joblib


def train_churn_model(csv_path, target_col='churn'):
    """
    Train a churn prediction model on the given CSV file.

    Requirements:
    - CSV must contain the target column (default: 'churn').
    - If 'customer_id' exists, it will be ignored as a feature.
    """

    # Read CSV
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f'Target column "{target_col}" not found in dataset')

    # 🔹 Drop target column + ID column (customer_id) from features
    drop_cols = [target_col]
    if 'customer_id' in df.columns:
        drop_cols.append('customer_id')

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Auto-detect numeric and categorical features
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns

    numeric_features = list(numeric_cols)
    categorical_features = list(cat_cols)

    # Preprocessor: passthrough numeric, OneHotEncode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    # Build pipeline: preprocessing + Logistic Regression model
    clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(max_iter=1000)),
        ]
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit model
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 🔹 Confusion matrix (TN, FP, FN, TP)
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback if something weird (multi-class etc.)
        tn = fp = fn = tp = 0

    # Save model to disk
    os.makedirs('saved_models', exist_ok=True)
    model_path = os.path.join('saved_models', 'churn_model.pkl')
    joblib.dump(clf, model_path)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'model_path': model_path,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }

    return metrics


def load_model(model_path):
    """
    Load a previously saved churn model from disk.
    """
    return joblib.load(model_path)
