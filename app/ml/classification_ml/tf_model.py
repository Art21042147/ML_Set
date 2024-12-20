import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


def train_tensorflow_classification(dataset_path, target_column, save_path):
    # Load dataset
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Encode target if categorical
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        np.save(f"{save_path}_label_encoder.npy", encoder.classes_)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    np.save(f"{save_path}_scaler.npy", scaler.mean_)
    np.save(f"{save_path}_scaler_std.npy", scaler.scale_)

    # Convert target to one-hot encoding
    y_train_one_hot = tf.keras.utils.to_categorical(y_train)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test)

    # Build TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(y_train_one_hot.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train_one_hot, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes)

    # Save model and metrics
    model.save(f"{save_path}.keras")
    with open(f"{save_path}_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(f"Classification Report:\n{report}")

    return accuracy, report


# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description="Run classification model")
parser.add_argument("--dataset-path", required=True, help="Path to the dataset")
parser.add_argument("--target-column", required=True, help="Target column for classification")
parser.add_argument("--save-path", required=True, help="Path to save results")
args = parser.parse_args()

# Запуск обучения
accuracy, report = train_tensorflow_classification(
    dataset_path=args.dataset_path,
    target_column=args.target_column,
    save_path=args.save_path
)

# Вывод результатов
print("Library: Tensorflow")
print(f"Dataset Path: {args.dataset_path}")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Results saved in: {args.save_path}")
