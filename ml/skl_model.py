from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd


class UniversalModel:
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier()
        self.scaler = StandardScaler()
        self.encoders = {}

    def preprocess(self, X):
        # Обработка числовых данных
        X_scaled = self.scaler.transform(X.select_dtypes(include=["float64", "int64"]))

        # Обработка категориальных данных
        X_categorical = X.select_dtypes(include=["object"])
        for col in X_categorical:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_categorical[col] = self.encoders[col].fit_transform(X_categorical[col])
            else:
                X_categorical[col] = self.encoders[col].transform(X_categorical[col])

        # Объединение обработанных данных
        return pd.concat([pd.DataFrame(X_scaled), X_categorical.reset_index(drop=True)], axis=1)

    def fit(self, X, y):
        # Сохранение метаинформации
        self.scaler.fit(X.select_dtypes(include=["float64", "int64"]))
        X_preprocessed = self.preprocess(X)
        self.model.fit(X_preprocessed, y)

    def predict(self, X):
        X_preprocessed = self.preprocess(X)
        return self.model.predict(X_preprocessed)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


# Использование
dataset = pd.read_csv("google_play_store_dataset.csv")
X, y = dataset.drop("target", axis=1), dataset["target"]

model = UniversalModel()
model.fit(X, y)

print(f"Accuracy: {model.evaluate(X, y):.2f}")
