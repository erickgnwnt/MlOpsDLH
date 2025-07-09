import os
import time
import joblib
import hydra
import numpy as np
from omegaconf import DictConfig
from data import DataProcessor
from model import LogisticModel, RandomForestModel
from model import KNNModel, OrdinalLogisticModel

# WANDB API Key for authentication
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if WANDB_API_KEY:
    import wandb
    wandb.login(key=WANDB_API_KEY)

np.float64 = np.float64  # Fixed NumPy 2.0 compatibility


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    start_time = time.time()

    # Data Processing
    data_processor = DataProcessor(cfg.data.path)
    X_train, X_test, y_train, y_test = data_processor.preprocess_data()

    # Save fitted scaler for inference
    scaler_path = os.path.join("models", "scaler.pkl")
    joblib.dump(data_processor.scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Initialize Models
    models = {
        "Logistic Regression": LogisticModel(),
        "Random Forest": RandomForestModel(
            n_estimators=cfg.random_forest.n_estimators,
            max_depth=cfg.random_forest.max_depth,
        ),
        "KNN": KNNModel(
            n_neighbors=(
                cfg.knn.n_neighbors if isinstance(cfg.knn.n_neighbors, int)
                else cfg.knn.n_neighbors[0]
            ),
            weights=(
                cfg.knn.weights if isinstance(cfg.knn.weights, str)
                else "uniform"
            ),
        ),
        "Ordinal Logistic Regression": OrdinalLogisticModel(),
    }

    # Train & Save Models
    for name, model in models.items():
        try:
            model.train(X_train, y_train)
            model_path = os.path.join(
                "models", f"{name.replace(' ', '_').lower()}.pkl"
            )
            joblib.dump(model.model, model_path)
            print(f"✅ Model saved: {model_path}")
            # Print model hyperparameters to console
            print(f"\n🔧 {name} Hyperparameters:")
            print(model.model.get_params())
        except Exception as e:
            print(f"⚠️ Error training {name}: {e}")

    # Evaluate Models in Parallel
    try:
        joblib.Parallel(n_jobs=-1)(
            joblib.delayed(m.evaluate)(
                X_train, X_test, y_train, y_test
            )
            for m in models.values()
        )
    except Exception as e:
        print(f"⚠️ Error during parallel evaluation: {e}")

    # Log execution time
    duration = time.time() - start_time
    print(f"🚀 Training completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
