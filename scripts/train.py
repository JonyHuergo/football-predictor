import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    """Load processed data and merge with outcomes"""
    df = pd.read_feather("data/processed/championship_processed.feather")
    raw_df = pd.read_csv("data/raw/E1.csv", encoding='utf-8-sig')
    df['FTR'] = raw_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    return df

def train_model():
    # Load and prepare data
    df = load_data()
    X = df.drop(columns=['FTR'])
    y = df['FTR']
    
    # Scale features (critical for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Initialize Logistic Regression with thesis parameters
    model = LogisticRegression(
        multi_class='multinomial',  # For 3-class prediction
        solver='lbfgs',             # Recommended for small datasets
        max_iter=1000,              # Ensure convergence
        class_weight='balanced',    # Handle imbalanced classes
        random_state=42
    )
    
    # Train with cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validated Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
    
    # Final training
    model.fit(X_train, y_train)
    
    # Evaluate
    print(f"\nTrain Accuracy: {accuracy_score(y_train, model.predict(X_train)):.2%}")
    print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
    
    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    
    # Save artifacts
    joblib.dump(model, "models/logistic_regression_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("\nModel and scaler saved to /models")

if __name__ == "__main__":
    train_model()