import numpy as np
from data import DataProcessor
from model import LogisticModel


class Inference:
    def __init__(self, data_path):
        self.data_processor = DataProcessor(data_path)
        self.model = LogisticModel()

        # Preprocess and train
        X_train, _, y_train, _ = self.data_processor.preprocess_data()
        self.model.train(X_train, y_train)

    def predict(self, new_data):
        new_data = np.array(new_data).reshape(1, -1)

        # Normalize using the fitted scaler
        scaler = self.data_processor.scaler
        new_data_scaled = scaler.transform(new_data)

        return self.model.predict(new_data_scaled)


if __name__ == "__main__":
    inference = Inference("data/raw/data.csv")
    new_sample = [50, 6.5, 3.2, 8.1, 0.7, 200, 1500]
    predicted_class = inference.predict(new_sample)
    print("Predicted Water Quality Status:", predicted_class)
