import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params=None):
        report = {}
        best_models = {}

        for model_name, model in models.items():
            try:
                # If params exist for this model → do hyperparameter tuning
                if params and model_name in params:
                    logging.info(f"Tuning {model_name} with GridSearchCV...")
                    gs = GridSearchCV(model, params[model_name], cv=3, n_jobs=-1, verbose=0, scoring="f1")
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)

                # Predictions
                y_pred = best_model.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                try:
                    auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
                except:
                    auc = 0.0

                report[model_name] = {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-score": f1,
                    "ROC-AUC": auc
                }

                best_models[model_name] = best_model

                logging.info(f"{model_name} trained successfully")

            except Exception as e:
                logging.error(f"Error training {model_name}: {e}")

        return report, best_models

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "SVM": SVC(probability=True),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "CatBoost": CatBoostClassifier(verbose=0)
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20]
                },
                "Random Forest": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_features": ["sqrt", "log2"],
                    "n_estimators": [50, 100, 200]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "subsample": [0.6, 0.8, 1.0],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10]
                }
            }

            # Evaluate with hyperparameter tuning
            model_report, best_models = self.evaluate_models(
                X_train, y_train, X_test, y_test, models=models, params=params
            )

            # Pick best model
            best_model_name = max(model_report, key=lambda k: model_report[k]["F1-score"])
            best_model_score = model_report[best_model_name]["F1-score"]
            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best model: {best_model_name} with F1-score: {best_model_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score, model_report

        except Exception as e:
            raise CustomException(e, sys)

