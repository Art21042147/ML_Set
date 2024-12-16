import pandas as pd
import numpy as np
import tensorflow as tf
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

# Example usage
datasets = {
    "pollution": {
        "path": "ml/datasets/processed_pollution_dataset.csv",
        "target_column": "Air Quality"
    },
    "energy": {
        "path": "ml/datasets/processed_renewable_energy.csv",
        "target_column": "Energy_Level"
    }
}

for name, details in datasets.items():
    print(f"Training TensorFlow classification model for {name}")
    accuracy, report = train_tensorflow_classification(
        dataset_path=details["path"],
        target_column=details["target_column"],
        save_path=f"ml/predictions/{name}_tensorflow_classification"
    )
    print(f"{name} - Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
