import numpy as np
from LogisticRegressionPU import LogisticRegressionPU


class UBPL: #Uncertainty-Based Pseudo-Labeling
    
    def __init__(self, K, J, t_l, t_u, verbose=0):
        
        self.t_l = t_l
        self.t_u = t_u
        self.K = K
        self.J = J
        self.verbose = verbose
    
    def fit(self, X, s):
        
        self.X = X
        self.n, self.m = X.shape
        self.s = s
        # pozycje etykiet:
        self.P = np.array(np.where(s==1.), dtype=int).flatten() # pozycje prawdziwych etykiet (równych 1)
        self.L = np.zeros(0, dtype=int) # pozycje pseudoetykiet
        self.U = np.array(np.where(s==0.), dtype=int).flatten() # pozycje obserwacji bez etykiet i bez pseudo-etykiet
    
        self.y = np.zeros(self.n) # etykiety, pseudoetykiety i 0 w przypadku braku obu
        self.y[self.P] = 1 # wstawiamy prawdziwe etykiety
        
        self.Proba = np.zeros(self.n) # prawdopodobieństwa/pseudo-etykiety

        D = np.concatenate((self.P, self.U)) # w pierwszym powtórzeniu bierzemy wszystkie obserwacje
        
        for j in range(self.J):

            if j>0:
                D = np.concatenate((self.P, self.L)) # w kolejnych powtórzeniach bierzemy obserwacje z etykietami i pseudoetykietami
                self.y[self.L] = self.Proba[self.L] # pseudoetykiety
                self.y[self.U] = 0 # brak etykiety - 0
                
            self.p_K = np.zeros((self.n, self.K)) # macierz na prawdopodobieństwa dla każdego powtórzenia
            
            for k in range(self.K):
                
                # próbka bootstrapowa
                D_baggy = np.random.choice(D, size=len(D)) 
                X_baggy = self.X.iloc[D_baggy,:]
                y_baggy = self.y[D_baggy]
                
                # regresja logistyczna
                model = LogisticRegressionPU(max_iter=15, tol=0.01, verbose=self.verbose)
                model.fit(X_baggy, y_baggy)
                self.p_K[:,k] = model.predict_proba(X)

            self.Proba = np.mean(self.p_K, axis=1) # uśredniamy prawdopodobieństwa dla każdej z n obserwacji, hat(p)_i
            
            for i in range(self.n):
                
                p_i = self.p_K[i,:] # hat(p)_ik
                
                # niepewności
                u_a = - np.mean(p_i * np.log(p_i + 1e-15) + (1-p_i) * np.log(1 - p_i + 1e-15))
                u_t = - self.Proba[i] * np.log(self.Proba[i] + 1e-15) - (1-self.Proba[i]) * np.log(1 - self.Proba[i] + 1e-15)
                u_e = u_t - u_a
                    
                # jeśli niepewność poniżej progu t_l, to mamy nową pseudoetykietę
                if u_e < self.t_l:
                    self.L = np.union1d(self.L, i)
                    self.U = np.setdiff1d(self.U, i)
                    
                # jeśli niepewność powyżej progu t_u, to usuwamy pseudoetykietę
                if u_e > self.t_u:
                    self.L = np.setdiff1d(self.L, i)
                    self.U = np.union1d(self.U, i)
                    
            # Usuwamy ze zbiorów L, U indeksy obserwacji poetykietowanych P
            self.L = np.setdiff1d(self.L, self.P)
            self.U = np.setdiff1d(self.U, self.P)
            
        return self
    
    def predict_proba(self):
        self.Proba[self.P] = 1 # prawdopodobieństwo 1 w przypadku etykiety 1
        return self.Proba
    
    def predict(self):
        self.Proba = self.predict_proba()
        return np.where(self.Proba>0.5, 1, 0)





