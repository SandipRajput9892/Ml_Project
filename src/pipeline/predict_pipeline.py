import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logging.info(">>>> Starting Prediction Pipeline <<<<")

        # Example: create a single-row dataframe with feature values
        sample_data = pd.DataFrame([{
            "Air temperature [K]": 300,
            "Process temperature [K]": 310,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 30,
            "Tool wear [min]": 100,
            "Type": "M"  # categorical column
        }])

        pipeline = PredictPipeline()
        prediction = pipeline.predict(sample_data)
        print("Prediction:", prediction)

    except Exception as e:
        raise CustomException(e, sys)
