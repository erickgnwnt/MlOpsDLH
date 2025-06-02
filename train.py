import wandb
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from data import DataProcessor
from model import LogisticModel


class Trainer:
    def __init__(self, data_path, project_name="logistic_regression_tuning"):
        wandb.init(project=project_name, name="Hyperparameter_Tuning", config={
            "model_type": "Logistic Regression",
            "initial_C": [0.01, 0.1, 1, 10, 100],
            "initial_solver": ["lbfgs", "saga", "newton-cg"],
            "initial_max_iter": [1000, 5000, 10000]
        })

        self.data_processor = DataProcessor(data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_processor.preprocess_data()

    def tune_hyperparameters(self):
        print("\nStarting Hyperparameter Tuning...")

        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],  
            "solver": ["lbfgs", "saga", "newton-cg"],  
            "max_iter": [1000, 5000, 10000]
        }

        grid_search = GridSearchCV(LogisticRegression(random_state=42), 
                                   param_grid, cv=5, scoring="accuracy", n_jobs=-1, return_train_score=True)

        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        wandb.config.update({
            "best_C": best_params["C"],
            "best_solver": best_params["solver"],
            "best_max_iter": best_params["max_iter"]
        })

        # Log hyperparameter tuning results correctly
        for i in range(len(grid_search.cv_results_["mean_test_score"])):
            wandb.log({
                "C": grid_search.cv_results_["param_C"][i],
                "Solver": str(grid_search.cv_results_["param_solver"][i]),  # Ensures WandB treats it as text
                "Max Iter": grid_search.cv_results_["param_max_iter"][i],
                "Validation Accuracy": grid_search.cv_results_["mean_test_score"][i]
            })

        print("Best Hyperparameters:", best_params)
        print("Best Accuracy:", grid_search.best_score_)

        return grid_search.best_estimator_

    def train_model(self, num_epochs=10):
        print("Training model...")
        start_time = time.time()

        self.model = self.tune_hyperparameters()

        for epoch in range(1, num_epochs + 1):
            self.model.fit(self.X_train, self.y_train)
            y_train_pred = self.model.predict(self.X_train)
            train_loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
            train_acc = accuracy_score(self.y_train, y_train_pred)

            # Log weight distributions to WandB
            weights = self.model.coef_
            biases = self.model.intercept_

            wandb.log({
                "Epoch": epoch,
                "Training Loss": train_loss,
                "Training Accuracy": train_acc,
                "Weight Distributions": wandb.Histogram(weights),
                "Bias Distributions": wandb.Histogram(biases)
            })

        end_time = time.time()
        wandb.log({"Training Time (seconds)": end_time - start_time})

        joblib.dump(self.model, "best_logistic_model.pkl")
        wandb.log_artifact("best_logistic_model.pkl", type="model")

        print("Final model saved as best_logistic_model.pkl")

    def evaluate_model(self):
        print("\nEvaluating model performance...")
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)

        wandb.log({"Training Accuracy": train_acc, "Test Accuracy": test_acc})
        wandb.log({"Classification Report": classification_report(self.y_test, y_test_pred, output_dict=True)})

        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')

        wandb.log({
            "CV Score Mean": cv_scores.mean(),
            "CV Score Std Dev": cv_scores.std()
        })

        for i, score in enumerate(cv_scores):
            wandb.log({f"CV Fold {i+1} Score": score})

        # Confusion Matrix Visualization
        conf_matrix = confusion_matrix(self.y_test, y_test_pred)
        fig, ax = plt.subplots()
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.7)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        wandb.log({"Confusion Matrix": wandb.Image(fig)})

        print("\nCross-validation scores:", cv_scores)
        print("Mean CV Accuracy:", cv_scores.mean())
        print("Standard Deviation:", cv_scores.std())

        wandb.finish()

# Run training with logging to WandB
trainer = Trainer("data.csv")
trainer.train_model()
trainer.evaluate_model()