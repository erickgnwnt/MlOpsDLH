import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
np.float_ = np.float64


class BaseModel:
    """Base model with automatic WandB logging."""

    def __init__(self, model, name, num_classes):
        self.model = model
        self.name = name
        self.num_classes = num_classes

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_train, X_test, y_train, y_test):
        wandb.finish()
        wandb.init(project="parallel_model_training", name=self.name)

        try:
            y_train_pred = self.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)

            y_test_pred = self.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=5, scoring="accuracy"
            )

            wandb.log({
                f"{self.name} Training Accuracy": train_acc,
                f"{self.name} Test Accuracy": test_acc,
                f"{self.name} CV Mean Score": cv_scores.mean(),
                f"{self.name} CV Score Std Dev": cv_scores.std(),
            })

            conf_matrix = confusion_matrix(y_test, y_test_pred)
            df_cm = pd.DataFrame(
                conf_matrix,
                columns=[f"Pred {i}" for i in range(len(conf_matrix))],
                index=[f"True {i}" for i in range(len(conf_matrix))]
            )

            plt.figure(figsize=(7, 4))
            sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d")
            wandb.log({
                f"{self.name} Confusion Matrix": wandb.Image(plt.gcf())
            })
            plt.close()

            wandb.log({
                f"{self.name} Classification Report": classification_report(
                    y_test, y_test_pred, output_dict=True
                )
            })

            if hasattr(self.model, "predict_proba"):
                y_probs = self.model.predict_proba(X_test)
                y_test_bin = label_binarize(
                    y_test, classes=list(range(self.num_classes))
                )

                pr_auc_scores = {}
                for i in range(self.num_classes):
                    precision, recall, _ = precision_recall_curve(
                        y_test_bin[:, i], y_probs[:, i]
                    )
                    pr_auc = auc(recall, precision)
                    pr_auc_scores[f"Class {i} PR AUC"] = pr_auc
                    plt.figure()
                    plt.plot(
                        recall,
                        precision,
                        marker=".",
                        label=f"AUC={pr_auc:.3f}"
                    )
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title(f"{self.name} Class {i} Precision-Recall Curve")
                    plt.legend()
                    plt.close()
                wandb.log(pr_auc_scores)

                roc_auc_scores = {}
                for i in range(self.num_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_auc_scores[f"Class {i} ROC AUC"] = roc_auc

                    plt.figure()
                    plt.plot(fpr, tpr, marker=".", label=f"AUC={roc_auc:.3f}")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"{self.name} Class {i} ROC Curve")
                    plt.legend()
                    wandb.log({
                        f"{self.name} Class {i} ROC": wandb.Image(plt.gcf())
                    })
                    plt.close()

                wandb.log(roc_auc_scores)

            if isinstance(self.model, RandomForestClassifier):
                feature_importance = pd.DataFrame({
                    "Feature": [
                        f"Feature {i}" for i in range(X_train.shape[1])
                    ],
                    "Importance": self.model.feature_importances_
                }).sort_values(by="Importance", ascending=False)

                wandb.log({
                    f"{self.name} Feature Importance": wandb.Table(
                        dataframe=feature_importance
                    )
                })

        except Exception as e:
            print(f"⚠️ Error during evaluation for {self.name}: {e}")

            wandb.log({
                f"{self.name} Evaluation Error": wandb.Html(
                    f"<pre>{str(e)}</pre>"
                )
            })
        wandb.finish()


class LogisticModel(BaseModel):
    def __init__(self, num_classes=3):
        super().__init__(
            LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
            "Logistic Regression", num_classes
        )

    def get_weights_bias(self):
        return self.model.coef_, self.model.intercept_


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, num_classes=3):
        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            ),
            "Random Forest", num_classes
        )


class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5, weights="uniform", num_classes=3):
        super().__init__(
            KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
            "KNN", num_classes
        )
