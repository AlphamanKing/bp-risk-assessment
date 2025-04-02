import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib
import os
from google.colab import files
import io

# Set random seed for reproducibility
np.random.seed(42)

def upload_dataset():
    """Prompt user to upload the dataset file"""
    print("Please upload the Hypertension-risk-model-main.csv file...")
    uploaded = files.upload()
    
    if not uploaded:
        raise ValueError("No file was uploaded. Please upload the dataset file.")
    
    # Read the uploaded file
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(io.StringIO(uploaded[file_name].decode('utf-8')))
    return df

def load_and_prepare_data():
    """Load and prepare the dataset"""
    # Load the dataset
    df = upload_dataset()
    
    # Define features and target
    features = [
        'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
        'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
        'heartRate', 'glucose'
    ]
    X = df[features]
    y = df['Risk']
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(0)  # Assuming missing values in target are 0 (no risk)
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def train_random_forest(X_train, y_train):
    """Train Random Forest with Grid Search"""
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Create base Random Forest model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1',
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    print("\nBest Hyperparameters:", grid_search.best_params_)
    print("Best F1 Score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def train_ensemble(X_train, y_train):
    """Train the ensemble model"""
    # Train individual models
    rf = train_random_forest(X_train, y_train)
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('lr', lr),
            ('svc', svc)
        ],
        voting='soft'
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    return ensemble

def evaluate_model(model, X, y, set_name):
    """Evaluate the model performance"""
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    # Print evaluation results
    print(f"\n--- {set_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

def save_and_download_models(ensemble, scaler):
    """Save models and trigger download"""
    import joblib
    import os
    from google.colab import files

    # Save models to disk
    joblib.dump(ensemble, 'ensemble_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Download files
    print("\nDownloading trained models...")
    files.download('ensemble_model.pkl')
    files.download('scaler.pkl')
    print("Models downloaded successfully!")


def main():
    print("Starting Blood Pressure Risk Assessment Model Training...")
    print("This script will:")
    print("1. Prompt you to upload the dataset")
    print("2. Train an ensemble model with SMOTE for class balancing")
    print("3. Evaluate the model on validation and test sets")
    print("4. Download the trained models\n")
    
    # Load and prepare data
    print("Loading and preparing data...")
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = load_and_prepare_data()
    
    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Train ensemble model
    print("\nTraining ensemble model...")
    ensemble = train_ensemble(X_train_balanced, y_train_balanced)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    evaluate_model(ensemble, X_val_scaled, y_val, "Validation Set")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluate_model(ensemble, X_test_scaled, y_test, "Test Set")
    
    # Save and download models
    save_and_download_models(ensemble, scaler)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 