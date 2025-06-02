from sklearn.linear_model import LogisticRegression

class LogisticModel:
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs',max_iter=1000,random_state=42)

    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
    
    def predict(self,X):
        return self.model.predict(X)
    
    def get_weights_bias(self):
        return self.model.coef_,self.model.intercept_