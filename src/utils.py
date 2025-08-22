import os
import sys
import dill
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, accuracy_score

def save_object(file_path, obj):
    """
    Save Python object using dill
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)


def load_object(file_path):
    """
    Load Python object using dill
    """
    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)


def evaluate_models(
    X_train, y_train, X_test, y_test,
    models: dict, params: dict,
    cv=3, n_jobs=-1, verbose=1, refit=True, scoring="r2"
):
    """
    Evaluate multiple models using GridSearchCV and return a detailed report.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")

            gs = GridSearchCV(
                model, params[model_name],
                cv=cv, n_jobs=n_jobs, verbose=verbose, refit=refit, scoring=scoring
            )
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Choose metric dynamically
            if scoring == "r2":
                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)
            elif scoring == "accuracy":
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)
            else:
                raise ValueError(f"Unsupported scoring metric: {scoring}")

            report[model_name] = {
                "train_score": train_score,
                "test_score": test_score,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
