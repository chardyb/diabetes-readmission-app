import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_model():
    print("--- Training Mode ---")

    #load data
    df = load_data('data/processed_data.csv')

    X = df.drop(columns=['target'])
    y = df['target']

    #define categorical & numerical columns
    categorical_cols = X.select_dtypes(include = ['object']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    print(f"Features: {X.shape[1]} total ({len(categorical_cols)} categorical, {len(numerical_cols)} numerical)")

    #Split data, stratifing on readmission (target variable)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

    #Preprocessing pipelines for both numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    #define model, XGBoost
    #weight is set to 8 to balance the classes
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    #combine preprocessing and modeling in a pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    #train
    print("Training model...")
    clf.fit(X_train, y_train)

    #evaluation
    print("\n--- Evaluation Metrics ---")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #save model 
    joblib.dump(clf, 'data/diabetes_model.pkl')
    print("Model saved to data/diabetes_model.pkl")

if __name__ == "__main__":
    train_model()