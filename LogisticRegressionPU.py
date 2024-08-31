import numpy as np 
import warnings
warnings.filterwarnings("ignore") 

# Model logistyczny dla pseudoetkiet z wykorzystaniem IRLS
class LogisticRegressionPU(): 
    
    def __init__(self, max_iter=15, tol=0.01, verbose=1):      
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
    def fit(self, X, Y):
        # n=liczba obserwacji, m=liczba parametrów (oprócz interceptu)        
        self.n, self.m = X.shape         
        # początkowe parametry      
        self.X = np.hstack((np.ones([self.n,1]), X))
        self.theta = np.zeros(self.m+1)
        self.Y = Y
        self.losses = []

        if len(np.unique(self.Y))==1: # zabezpieczenie - co najmniej 2 klasy
            print('''LogisticRegressionPU needs samples of at least 2 classes in the data,
but the data contains only one class: {i}'''.format(i=np.unique(self.Y)[0]))
            return self

        # IRLS
        for i in range(self.max_iter):
 
            self.z = self.working_response(self.X, self.Y, self.theta)
            self.theta = self.update_parameters(self.X, self.W, self.z)            
            self.compute_loss(self.Y, self.p, self.theta)
            
            if i>1:
                if abs(self.losses[-1]-self.losses[-2])<self.tol:
                    self.coefs()
                    self.converged=i+1
                    if self.verbose != 0:
                        print('IRLS converged in {i} iterations.'.format(i=self.converged))
                    break
                    
            if i==self.max_iter-1:
                if self.verbose != 0:
                    print('Warning: IRLS failed to converge. Try increasing the number of iterations.')
        
        self.coefs()   
        return self
    
    def coefs(self):
        self.coef_ = self.theta[1:]
        self.intercept_ = self.theta[0]
        
    def working_response(self, X, Y, theta):
        self.p = self.sigmoid(X @ theta)
        self.p[self.p==1]=1-1e-9
        self.p[self.p==0]=1e-9
        self.W = np.diag(self.p * (1-self.p))
        return X @ theta + np.linalg.inv(self.W) @ (Y-self.p) 
    
    def update_parameters(self, X, W, z):
        return np.linalg.inv(X.T @ W @ X) @ X.T @ W @ z
        
    def compute_loss(self, Y, pred, theta):
        loss=-np.mean(Y * np.log(pred + 1e-9)+ (1-Y) * np.log(1 - pred + 1e-9))
        self.losses += [loss]
    
    def sigmoid(self, v):
        return np.array([self.sigmoid_function(value) for value in v])
    
    def sigmoid_function(self, z):
        if z>=0:
            t=np.exp(-z)
            return 1/(1+t)
        else:
            t=np.exp(z)
            return t/(1+t)
        
    def predict_proba(self, X):
        X = np.hstack(((np.ones([X.shape[0],1]), X)))
        return self.sigmoid(X @ self.theta)
    
    def predict(self, X):
        pred = self.predict_proba(X)        
        Y_pred = np.where(pred>0.5, 1, 0)         
        return Y_pred