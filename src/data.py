import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if WANDB_API_KEY:
    import wandb
    wandb.login(key=WANDB_API_KEY)


class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.features = [
            "TSS", "DO", "BOD", "COD", "Fosfat", "Fecal Coli", "Total-Coliform"
        ]
        self.target = "WaterQualityStatus_encoded"
        self.scaler = StandardScaler()

    def load_data(self):
        """Load and clean dataset."""
        try:
            df = pd.read_csv(self.filepath)

            # Convert features & target to numeric, handling errors
            df[self.features] = df[self.features].apply(
                pd.to_numeric, errors="coerce"
            )
            df[self.target] = pd.to_numeric(df[self.target], errors="coerce")

            # Log missing values
            missing_values = df.isnull().sum()
            print("🔍 Missing values before cleaning:\n", missing_values)

            # Drop rows with missing target or features
            df_clean = df.dropna(subset=self.features + [self.target])

            print(f"✅ Loaded dataset with {len(df_clean)} samples.")
            return df_clean[self.features], df_clean[self.target]

        except Exception as e:
            print(f"🚨 Error loading data: {e}")
            return None, None

    def preprocess_data(self):
        """Preprocess data, split, and save scaled values."""
        X, y = self.load_data()
        if X is None or y is None:
            print("🚨 Data loading failed. Returning None.")
            return None, None, None, None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        os.makedirs("data/processed", exist_ok=True)

        df_train = pd.DataFrame(X_train_scaled, columns=self.features)
        df_train[self.target] = y_train
        df_train.to_csv("data/processed/train.csv", index=False)

        df_test = pd.DataFrame(X_test_scaled, columns=self.features)
        df_test[self.target] = y_test
        df_test.to_csv("data/processed/test.csv", index=False)

        print(
            f"✅ Preprocessed data saved:\n"
            f" - Train: train.csv ({len(df_train)})\n"
            f" - Test: data/processed/test.csv "
            f"({len(df_test)} samples)"
        )

        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    processor = DataProcessor("data/raw/data.csv")
    processor.preprocess_data()
