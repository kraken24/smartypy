"""
Prompt: 
Write unit tests in pytest for the code in <context_file>.
Also write proper docstrings and type hinting for all test functions.
"""

# BUG: certain import statements are missing in the generated code

import pytest
import IrisClassification # need to fix the import statement


@pytest.fixture
def iris_classification():
    return IrisClassification()

def test_preprocess_data(iris_classification):
    """
    Test the preprocess_data method.
    
    Args:
        iris_classification (IrisClassification): The IrisClassification object to be tested.
    
    Returns:
        None
    """
    iris_classification.preprocess_data()
    assert isinstance(iris_classification.data, pd.DataFrame)
    assert isinstance(iris_classification.target, pd.Series)

def test_split_data(iris_classification):
    """
    Test the split_data method.
    
    Args:
        iris_classification (IrisClassification): The IrisClassification object to be tested.
    
    Returns:
        None
    """
    X_train, X_test, y_train, y_test = iris_classification.split_data()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model(iris_classification):
    """
    Test the train_model method.
    
    Args:
        iris_classification (IrisClassification): The IrisClassification object to be tested.
    
    Returns:
        None
    """
    X_train = iris_classification.data
    y_train = iris_classification.target
    model = iris_classification.train_model(X_train, y_train)
    assert isinstance(model, XGBClassifier)

def test_evaluate_model(iris_classification):
    """
    Test the evaluate_model method.
    
    Args:
        iris_classification (IrisClassification): The IrisClassification object to be tested.
    
    Returns:
        None
    """
    model = iris_classification.train_model(iris_classification.data, iris_classification.target)
    X_test = iris_classification.data
    y_test = iris_classification.target
    iris_classification.evaluate_model(model, X_test, y_test)

def test_save_model(iris_classification):
    """
    Test the save_model method.
    
    Args:
        iris_classification (IrisClassification): The IrisClassification object to be tested.
    
    Returns:
        None
    """
    model = iris_classification.train_model(iris_classification.data, iris_classification.target)
    iris_classification.save_model(model)

def test_load_model(iris_classification):
    """
    Test the load_model method.
    
    Args:
        iris_classification (IrisClassification): The IrisClassification object to be tested.
    
    Returns:
        None
    """
    model = iris_classification.train_model(iris_classification.data, iris_classification.target)
    iris_classification.save_model(model)
    loaded_model = iris_classification.load_model()
    assert isinstance(loaded_model, XGBClassifier)

def test_make_predictions(iris_classification):
    """
    Test the make_predictions method.
    
    Args:
        iris_classification (IrisClassification): The IrisClassification object to be tested.
    
    Returns:
        None
    """
    model = iris_classification.train_model(iris_classification.data, iris_classification.target)
    X_new = iris_classification.data
    predictions = iris_classification.make_predictions(model, X_new)
    assert isinstance(predictions, pd.Series)

if __name__ == "__main__":
    pytest.main([str(__file__)])
