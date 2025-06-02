import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class DataProcessor:

    def __init__(self,filepath):
        self.filepath = filepath
        self.features = ['TSS', 'DO', 'BOD', 'COD', 'Fosfat', 'Fecal Coli', 'Total-Coliform']
        self.target = 'WaterQualityStatus_encoded'
        self.scaler = StandardScaler()
                         
    def load_data(self):
        df = pd.read_csv(self.filepath)

        for col in self.features:
            df[col] = pd.to_numeric(df[col],errors='coerce')
        
        df[self.target] = pd.to_numeric(df[self.target],errors='coerce')
        df_clean = df.dropna(subset=self.features + [self.target])

        return df_clean[self.features], df_clean[self.target]
    
    def preprocess_data(self):
        X, y = self.load_data()
        X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    
        


