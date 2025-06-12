import os
import time
import joblib
import numpy as np
import hydra
from omegaconf import DictConfig

from data import DataProcessor
from model import LogisticModel, RandomForestModel, KNNModel

np.float_ = np.float64


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    start_time = time.time()

    # Data Processing
    data_processor = DataProcessor(cfg.data.path)
    X_train, X_test, y_train, y_test = data_processor.preprocess_data()

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Initialize Models
    models = {
        "Logistic Regression": LogisticModel(),
        "Random Forest": RandomForestModel(
            n_estimators=cfg.random_forest.n_estimators,
            max_depth=cfg.random_forest.max_depth
        ),
        "KNN": KNNModel(
            n_neighbors=cfg.knn.n_neighbors
            if isinstance(cfg.knn.n_neighbors, int)
            else cfg.knn.n_neighbors[0],
            weights=cfg.knn.weights
            if isinstance(cfg.knn.weights, str)
            else "uniform"
        ),
    }

    # Train & Save Models
    for name, model in models.items():
        model.train(X_train, y_train)

        # Save trained model to disk
        model_path = f"models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model.model, model_path)
        print(f"✅ Model saved: {model_path}")

    # Evaluate Models in Parallel
    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(m.evaluate)(X_train, X_test, y_train, y_test)
        for m in models.values()
    )

    # Log execution time
    duration = time.time() - start_time
    print(f"🚀 Training completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()