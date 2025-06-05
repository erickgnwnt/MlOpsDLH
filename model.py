import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

class BaseModel:
    """Base model with automatic WandB logging."""
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_train, X_test, y_train, y_test):
        # Ensure previous WandB run is closed before starting a new one
        wandb.finish()  
        wandb.init(project="parallel_model_training", name=self.name)

        try:
            y_train_pred = self.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)

            y_test_pred = self.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="accuracy")

            # Log key metrics to WandB
            wandb.log({
                f"{self.name} Training Accuracy": train_acc,
                f"{self.name} Test Accuracy": test_acc,
                f"{self.name} CV Mean Score": cv_scores.mean(),
                f"{self.name} CV Score Std Dev": cv_scores.std(),
            })

            # Log confusion matrix
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            df_cm = pd.DataFrame(conf_matrix, columns=[f"Pred {i}" for i in range(len(conf_matrix))], 
                                 index=[f"True {i}" for i in range(len(conf_matrix))])
            
            plt.figure(figsize=(7, 4))
            sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d")
            wandb.log({f"{self.name} Confusion Matrix": wandb.Image(plt.gcf())})

            # Log classification report
            wandb.log({f"{self.name} Classification Report": classification_report(y_test, y_test_pred, output_dict=True)})

            # Properly handle feature importance for RandomForest
            if isinstance(self.model, RandomForestClassifier):
                feature_importance = pd.DataFrame({
                    "Feature": [f"Feature {i}" for i in range(X_train.shape[1])],  # Use indices instead of column names
                    "Importance": self.model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                wandb.log({f"{self.name} Feature Importance": wandb.Table(dataframe=feature_importance)})

        except Exception as e:
            print(f"⚠️ Error during evaluation for {self.name}: {e}")
            wandb.log({f"{self.name} Evaluation Error": wandb.Html(f"<pre>{str(e)}</pre>")})  # Logs error properly as text

        wandb.finish()  # Properly close WandB session after logging

class LogisticModel(BaseModel):
    def __init__(self):
        super().__init__(LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42), "Logistic Regression")

    def get_weights_bias(self):
        return self.model.coef_, self.model.intercept_

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42), "Random Forest")

class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5, weights="uniform"):
        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights), "KNN")