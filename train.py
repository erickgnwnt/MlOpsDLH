import hydra
from omegaconf import DictConfig
import joblib
import time
from data import DataProcessor
from model import LogisticModel, RandomForestModel, KNNModel  # Corrected imports

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    start_time = time.time()  # Start timer

    # Data Processing
    data_processor = DataProcessor(cfg.data.path)
    X_train, X_test, y_train, y_test = data_processor.preprocess_data()

    # Initialize Models
    models = {
        "Logistic Regression": LogisticModel(),
        "Random Forest": RandomForestModel(n_estimators=cfg.random_forest.n_estimators, max_depth=cfg.random_forest.max_depth),
        "KNN": KNNModel(n_neighbors=cfg.knn.n_neighbors if isinstance(cfg.knn.n_neighbors, int) else cfg.knn.n_neighbors[0], 
                         weights=cfg.knn.weights if isinstance(cfg.knn.weights, str) else "uniform"),
    }

    # 🔹 Train Models First
    for model in models.values():
        model.train(X_train, y_train)

    # 🔹 Evaluate Models in Parallel
    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(lambda model: model.evaluate(X_train, X_test, y_train, y_test))(model)
        for model in models.values()
    )

    # Log execution time
    end_time = time.time()
    print(f"🚀 Training completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()