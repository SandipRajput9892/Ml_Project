import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info(">>>> Starting Training Pipeline <<<<")

        # 1. Data Ingestion
        ingestion_obj = DataIngestion()
        train_path, test_path = ingestion_obj.initiate_data_ingestion()

        # 2. Data Transformation
        transformation_obj = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation_obj.initiate_data_transformation(
            train_path, test_path
        )

        # Split features (X) and target (y)
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        # 3. Model Training
        model_trainer = ModelTrainer()
        best_model_name, best_model_score, model_report = model_trainer.initiate_model_trainer(
            X_train, y_train, X_test, y_test
        )

        logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
        logging.info(">>>> Training Pipeline Completed Successfully <<<<")

    except Exception as e:
        raise CustomException(e, sys)
