import os
import joblib
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_classification_model(dataset, target_column, save_path):
    """
    Train a classification model using scikit-learn and save results.

    :param dataset: pandas DataFrame containing the dataset.
    :param target_column: str, the name of the target column for classification.
    :param save_path: str, path to save the model, metrics, and plots.
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Split data into features and target
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Encode target if categorical
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        joblib.dump(encoder, os.path.join(save_path, 'label_encoder.pkl'))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, os.path.join(save_path, 'classification_model.pkl'))

    # Predictions and metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save metrics
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        f.write(f'Accuracy: {accuracy}\n\n')
        f.write(f'Classification Report:\n{report}\n\n')
        f.write(f'Confusion Matrix:\n{conf_matrix}\n')

    # Generate and save plots
    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix')
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

    return accuracy, report


# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description="Run classification model")
parser.add_argument("--dataset-path", required=True, help="Path to the dataset")
parser.add_argument("--target-column", required=True, help="Target column for classification")
parser.add_argument("--save-path", required=True, help="Path to save results")
args = parser.parse_args()

# Загрузка датасета
dataset = pd.read_csv(args.dataset_path)

# Запуск обучения
accuracy, report = train_classification_model(
    dataset=dataset,
    target_column=args.target_column,
    save_path=args.save_path
)

# Вывод результатов
print("Library: Scikit-learn")
print(f"Dataset Path: {args.dataset_path}")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Results saved in: {args.save_path}")
