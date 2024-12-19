import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_regression_model(dataset_path, target_column, save_path):
    """
    Train a regression model using Random Forest and save results.

    :param dataset_path: str, path to the dataset CSV file.
    :param target_column: str, the name of the target column for regression.
    :param save_path: str, directory to save the model, metrics, and plots.
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load dataset
    dataset = pd.read_csv(dataset_path)

    # Split data into features and target
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Encode categorical features if needed
    for col in X.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, os.path.join(save_path, 'regression_model.pkl'))

    # Predictions and metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save metrics
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        f.write(f'MSE: {mse}\n')
        f.write(f'R²: {r2}\n')

    # Generate and save feature importance plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importances.png'))
    plt.close()

    return mse, r2


# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run regression model")
parser.add_argument("--dataset-path", required=True, help="Path to the dataset")
parser.add_argument("--target-column", required=True, help="Target column for regression")
parser.add_argument("--save-path", required=True, help="Path to save results")
args = parser.parse_args()

# Pass arguments to the function
mse, r2 = train_regression_model(
    dataset_path=args.dataset_path,
    target_column=args.target_column,
    save_path=args.save_path
)

# Output results
print(f"Dataset Path: {args.dataset_path}")
print(f"MSE: {mse}")
print(f"R²: {r2}")
print(f"Results saved in: {args.save_path}")
