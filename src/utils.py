import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Saves the given object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Saving object to {file_path}")
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e
    
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates the given models and returns a report with their performance metrics.
    """
    
    report = {}
    
    for i in range(len(list(models))):
        model = list(models.values())[i]
        para=param[list(models.keys())[i]]

        gs = GridSearchCV(model,para,cv=3)
        gs.fit(X_train,y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train,y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)
        report[list(models.keys())[i]] = test_model_score
        
    return report


def load_object(file_path):
    """
    Loads an object from a file using pickle.
    """
    try:
        logging.info(f"Loading object from {file_path}")
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys) from e