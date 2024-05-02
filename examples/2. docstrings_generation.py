"""
Prompt: 
For the code above, write docstrings for all the methods and classes. Provide
the necessary type-hinting for the input and output arguments.
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class IrisClassification:
    """
    Class for iris classification using XGBoost.
    
    Attributes:
        iris (sklearn.datasets.load_iris): The iris dataset.
        data (pandas.DataFrame): The preprocessed iris data.
        target (pandas.Series): The iris target values.
    """

    def __init__(self) -> None:
        """
        Initializes the IrisClassification object by loading the iris dataset and preprocessing the data.
        
        Returns:
            None
        """
        self.iris = load_iris()
        self.data = pd.DataFrame(data=self.iris.data, columns=self.iris.feature_names)
        self.target = pd.Series(self.iris.target)

    def preprocess_data(self) -> None:
        """
        Preprocesses the iris data by applying imputation, scaling, and one-hot encoding.
        
        Returns:
            None
        """
        preprocessing_steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))
        ]
        column_transformer = ColumnTransformer(
            transformers=preprocessing_steps,
            remainder='passthrough'
        )
        self.data = column_transformer.fit_transform(self.data)

    def split_data(self) -> tuple:
        """
        Splits the iris data into training and testing sets.
        
        Returns:
            X_train (numpy.array), X_test (numpy.array), y_train (numpy.array), y_test (numpy.array)
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """
        Trains an XGBoost classifier on the training data.
        
        Args:
            X_train (pandas.DataFrame): The training features.
            y_train (pandas.Series): The training targets.
        
        Returns:
            model (XGBClassifier): The trained XGBoost classifier.
        """
        model = XGBClassifier()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Evaluates the performance of an XGBoost classifier on the testing data.
        
        Args:
            model (XGBClassifier): The trained XGBoost classifier.
            X_test (pandas.DataFrame): The testing features.
            y_test (pandas.Series): The testing targets.
        
        Returns:
            None
        """
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        confusion_matrix_val = confusion_matrix(y_test, predictions)
        print(f"Accuracy: {accuracy:.3f}")
        print("Classification Report:\n", report)
        plt.figure(figsize=(10, 6))
        plt.plot(confusion_matrix_val.flatten())
        plt.xlabel('Confusion Matrix')
        plt.ylabel('Count')
        plt.show()

    def save_model(self, model: XGBClassifier) -> None:
        """
        Saves an XGBoost classifier to a file.
        
        Args:
            model (XGBClassifier): The XGBoost classifier to be saved.
        
        Returns:
            None
        """
        import joblib
        joblib.dump(model, 'iris_classification_model.joblib')

    def load_model(self) -> XGBClassifier:
        """
        Loads an XGBoost classifier from a file.
        
        Returns:
            model (XGBClassifier): The loaded XGBoost classifier.
        """
        import joblib
        return joblib.load('iris_classification_model.joblib')

    def make_predictions(self, model: XGBClassifier, X_new: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using an XGBoost classifier.
        
        Args:
            model (XGBClassifier): The trained XGBoost classifier.
            X_new (pandas.DataFrame): The new features to be predicted.
        
        Returns:
            predictions (pandas.Series): The predicted targets.
        """
        predictions = model.predict(X_new)
        return predictions

if __name__ == "__main__":
    iris_classification = IrisClassification()
    iris_classification.preprocess_data()
    X_train, X_test, y_train, y_test = iris_classification.split_data