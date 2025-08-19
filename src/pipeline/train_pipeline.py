import sys
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

        # 3. Model Training
        trainer_obj = ModelTrainer()
        trainer_obj.initiate_model_training(train_arr, test_arr)


        logging.info(">>>> Training Pipeline Completed Successfully <<<<")

    except Exception as e:
        raise CustomException(e, sys)
