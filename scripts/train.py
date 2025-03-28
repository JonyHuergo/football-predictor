import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from preprocess import FootballDataPreprocessor

def load_data():
    """Load multiple seasons of data with correct FIFA ratings"""
    season_map = {
        '2021-2022': {
            'matches': 'data/raw/E1_2021_2022.csv',
            'processed': 'data/processed/E1_2021_2022.feather',
            'fifa': 'data/raw/FIFA_2022_RATINGS.csv'
        },
        '2022-2023': {
            'matches': 'data/raw/E1_2022_2023.csv',
            'processed': 'data/processed/E1_2022_2023.feather',
            'fifa': 'data/raw/FIFA_2023_RATINGS.csv'
        },
        '2023-2024': {
            'matches': 'data/raw/E1_2023_2024.csv',
            'processed': 'data/processed/E1_2023_2024.feather',
            'fifa': 'data/raw/FIFA_2024_RATINGS.csv'
        },
        '2024-2025': {
            'matches': 'data/raw/E1_2024_2025.csv',
            'processed': 'data/processed/E1_2024_2025.feather',
            'fifa': 'data/raw/FIFA_2025_RATINGS.csv'
        }
    }
    
    all_data = []
    for season_name, paths in season_map.items():
        # Load and process each season with its own FIFA ratings
        raw_df = pd.read_csv(paths['matches'], encoding='utf-8-sig')
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], dayfirst=True)
        raw_df['Season'] = season_name
        
        # Process with season-specific FIFA ratings
        processor = FootballDataPreprocessor(fifa_ratings_path=paths['fifa'])
        processed_df = processor.transform(raw_df)
        processed_df['Season'] = season_name
        processed_df['FTR'] = raw_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        
        all_data.append(processed_df)
    
    full_df = pd.concat(all_data).dropna()
    
    # Split into train (first 3 seasons) and test (current season)
    train_df = full_df[~full_df['Season'].str.contains('2024-2025')]
    test_df = full_df[full_df['Season'].str.contains('2024-2025')]
    
    return train_df, test_df

def train_model():
    # Load and split data by season
    train_df, test_df = load_data()
    
    # Prepare features and targets
    X_train = train_df.drop(columns=['FTR', 'Season'])
    y_train = train_df['FTR']
    X_test = test_df.drop(columns=['FTR', 'Season'])
    y_test = test_df['FTR']
    
    # Check class distribution
    print("\nTraining Class Distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nTest Class Distribution:")
    print(y_test.value_counts(normalize=True))
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Initialize model
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    # Cross-validation on training data only
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validated Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print(f"\nTrain Accuracy: {accuracy_score(y_train, model.predict(X_train_scaled)):.2%}")
    print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test_scaled)):.2%}")
    
    # Classification report
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, model.predict(X_test_scaled)))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.mean(np.abs(model.coef_), axis=0)
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Save artifacts
    joblib.dump(model, "models/logistic_regression_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("\nModel and scaler saved to /models")

if __name__ == "__main__":
    train_model()