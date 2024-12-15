import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(filepath, target_column):
    """
    Preprocess a dataset for machine learning.

    :param filepath: str, path to the dataset file (CSV).
    :param target_column: str, name of the target column for classification or regression.
    :return: pandas DataFrame, processed dataset.
    """
    # Load the dataset
    dataset = pd.read_csv(filepath)

    # Handle missing values (example: fill numeric columns with mean)
    for column in dataset.columns:
        if dataset[column].dtype in ['float64', 'int64']:
            dataset[column].fillna(dataset[column].mean(), inplace=True)
        elif dataset[column].dtype == 'object':
            dataset[column].fillna(dataset[column].mode()[0], inplace=True)

    # Encode categorical columns
    for column in dataset.select_dtypes(include=['object']).columns:
        if column != target_column:
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])

    # Process the target column
    if dataset[target_column].dtype == 'object':
        le = LabelEncoder()
        dataset[target_column] = le.fit_transform(dataset[target_column])

    return dataset
