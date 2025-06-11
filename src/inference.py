import numpy as np
from data import DataProcessor
from model import LogisticModel

class Inference:
    def __init__(self, data_path):
        self.data_processor = DataProcessor(data_path)
        self.model = LogisticModel()
        self.X_train, _, _, _ = self.data_processor.preprocess_data()
        self.model.train(self.X_train, self.data_processor.load_data()[1])  # Melatih model dengan data yang tersedia

    def predict(self, new_data):
        new_data = np.array(new_data).reshape(1, -1)

        # Normalisasi dengan scaler yang sudah dipakai sebelumnya
        scaler = self.data_processor.scaler
        new_data_scaled = scaler.transform(new_data)

        return self.model.predict(new_data_scaled)

# Contoh prediksi data baru
inference = Inference("data.csv")
new_sample = [50, 6.5, 3.2, 8.1, 0.7, 200, 1500]
predicted_class = inference.predict(new_sample)
print("Predicted Water Quality Status:", predicted_class)