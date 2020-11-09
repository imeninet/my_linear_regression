
class MyLinearRegression:
    
    def __init__(self):
        self.w = None
        pass
    
    def fit(self, X, y):
        a = np.ones([len(X) , 1])
        X = np.concatenate( [X , a], 1) 
        self.w = np.linalg.inv(X.T@X)@X.T@y
        
    def predict(self, X):
        a = np.ones([len(X) , 1])
        X = np.concatenate( [X , a], 1) 
        
        y_pred = X@self.w
        return y_pred
    
    def get_weights(self):
        return self.w

