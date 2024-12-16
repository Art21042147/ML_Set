import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class EnhancedRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_pytorch_regression_with_early_stopping(dataset_path, target_column, save_path):
    # Load dataset
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns=[target_column]).values
    y = dataset[target_column].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    model = EnhancedRegressionModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # Training loop with early stopping
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(300):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # Validate model
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).numpy()
            val_loss = mean_squared_error(y_test.numpy(), y_pred)

        # Scheduler step
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_path}_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        mse = mean_squared_error(y_test.numpy(), y_pred)
        r2 = r2_score(y_test.numpy(), y_pred)

    # Save metrics
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
    print(f"Training Enhanced PyTorch regression model with Early Stopping for {name}")
    mse, r2 = train_pytorch_regression_with_early_stopping(
        dataset_path=details["path"],
        target_column=details["target_column"],
        save_path=f"ml/predictions/{name}_early_stopping_pytorch_regression"
    )
    print(f"{name} - MSE: {mse}, R²: {r2}")
