"""
Prompt: 
Write a Python script that accomplishes the above mentioned to-dos.
Make sure to properly organise the code into functions and classes to improve readability.
"""
# TODO: Import necessary libraries


# TODO: Write a function to load the iris dataset from scikit-learn


# TODO: Write a function to apply data preprocessing using scikit-learn ColumnTransformer to check for missing values, outlier removal, data normalisation etc.


# TODO: Write a function to split data into training and testing sets


# TODO: Write a function to train a basic classification model using XgBoost


# TODO: Write a function to wrap the data preprocessing and training using scikit-learn Pipeline for 


# TODO: Write a function to evaluate the trained model


# TODO: Write a function to display relevant classification metrics and plot the performance of the model


# TODO: Write a function to save & load the model pipeline


# TODO: Write a function to use the model pipeline to generate predictions using unseen data

# ==== Llama 3 Generated Code =================================================
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
    def __init__(self):
        self.iris = load_iris()
        self.data = pd.DataFrame(data=self.iris.data, columns=self.iris.feature_names)
        self.target = pd.Series(self.iris.target)

    def preprocess_data(self):
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

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        model = XGBClassifier()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
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

    def save_model(self, model):
        import joblib
        joblib.dump(model, 'iris_classification_model.joblib')

    def load_model(self):
        import joblib
        return joblib.load('iris_classification_model.joblib')

    def make_predictions(self, model, X_new):
        predictions = model.predict(X_new)
        return predictions

if __name__ == "__main__":
    iris_classification = IrisClassification()
    iris_classification.preprocess_data()
    X_train, X_test, y_train, y_test = iris_classification.split_data()
    model = iris_classification.train_model(X_train, y_train)
    iris_classification.evaluate_model(model, X_test, y_test)
    iris_classification.save_model(model)
    loaded_model = iris_classification.load_model()
    new_data = pd.DataFrame(data=[[5.1, 3.5, 1.4, 0.2]], columns=iris_classification.data.columns)
    predictions = iris_classification.make_predictions(loaded_model, new_data)
    print("Predictions:", predictions)
