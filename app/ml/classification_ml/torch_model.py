import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


class ClassificationModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Increased neurons
        self.fc2 = nn.Linear(128, 64)  # Added complexity
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_pytorch_classification(dataset_path, target_column, save_path):
    # Load dataset
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns=[target_column]).values
    y = dataset[target_column].values

    # Encode target if categorical
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        torch.save(encoder.classes_, f"{save_path}_label_encoder.pt")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    torch.save({'mean': scaler.mean_, 'scale': scaler.scale_}, f"{save_path}_scaler.pt")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Initialize model, loss, and optimizer
    model = ClassificationModel(input_dim=X_train.shape[1], num_classes=len(torch.unique(y_train)))
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted loss
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate

    # Training loop
    for epoch in range(200):  # Increased epochs
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_classes = torch.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes)

    # Save model and metrics
    torch.save(model.state_dict(), f"{save_path}_model.pth")
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
accuracy, report = train_pytorch_classification(
    dataset_path=args.dataset_path,
    target_column=args.target_column,
    save_path=args.save_path
)

# Вывод результатов
print("Library: Pytorch")
print(f"Dataset Path: {args.dataset_path}")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Results saved in: {args.save_path}")
