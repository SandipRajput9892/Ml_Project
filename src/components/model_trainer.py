import os
import sys
import pickle
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Try two models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
            }

            best_model = None
            best_score = 0
            best_model_name = None

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                logging.info(f"{name} Accuracy: {acc}")

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_model_name = name

            logging.info(f"Best Model: {best_model_name} with Accuracy: {best_score}")

            # Save the best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Run full pipeline
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    best_model_name, best_score = trainer.initiate_model_training(train_arr, test_arr)

    print(f"✅ Best Model: {best_model_name} with Accuracy: {best_score}")
