import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_tensorflow_regression(dataset_path, target_column, save_path):
    # Load dataset
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)

    # Evaluate model
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model and metrics
    model.save(f"{save_path}.keras")  # Сохраняем в формате .keras
    with open(f"{save_path}_metrics.txt", "w") as f:
        f.write(f"MSE: {mse}\nR²: {r2}\n")

    return mse, r2

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
    print(f"Training TensorFlow regression model for {name}")
    mse, r2 = train_tensorflow_regression(
        dataset_path=details["path"],
        target_column=details["target_column"],
        save_path=f"ml/predictions/{name}_tensorflow_regression"
    )
    print(f"{name} - MSE: {mse}, R²: {r2}")
