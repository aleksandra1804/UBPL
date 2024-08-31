import numpy as np

# Funkcja do tworzenia danych PU dla zadanego c i scenariusza S1 lub S2 
 
def make_pu_labels(X, y, prob_true, label_scheme='S1', c=0.5): 

    if prob_true.shape[0]!=X.shape[0]:
        raise Exception('The length of prob_true does not match the number of instances in X')
    prob_true[np.where(prob_true==1)] = 0.999
    prob_true[np.where(prob_true==0)] = 0.001
    n = X.shape[0]
    s = np.zeros(n)
    ex_true = np.zeros(n)    
    if label_scheme=='S1':
        for i in np.arange(0,n,1):
            ex_true[i] = c
    elif label_scheme=='S2':
        if any(prob_true)==None:
            raise Exception('Argument prob_true should be specified')           
        lin_pred = np.log(prob_true/(1-prob_true))  
        a_seq = np.linspace(start=-10, stop=10, num=100)
        score = np.zeros(100)
        k=0
        w1 = np.where(y==1)[0]
        for a in a_seq:
            for i in np.arange(0,n,1):
                ex_true[i] = sigmoid_function(lin_pred[i] + a)
            score[k] = np.abs(np.mean(ex_true[w1])-c)
            k=k+1
        a_opt = a_seq[np.argmin(score)]
        for i in np.arange(0,n,1):
            ex_true[i] = sigmoid_function(lin_pred[i] + a_opt)
    else:
        print('Argument label_scheme is not defined')
    for i in np.arange(0,n,1):
        if y[i]==1:
            s[i]=np.random.binomial(1, ex_true[i], size=1)
    if np.sum(s)<=1:
        s[np.random.choice(s.shape[0],2,replace=False)]=1
        warnings.warn('Warning: <2 observations with s=1. Two random instances were assigned label s=1.')
    return s, ex_true

def sigmoid_function(z):
    if z>=0:
        t=np.exp(-z)
        return 1/(1+t)
    else:
        t=np.exp(z)
        return t/(1+t)
        
def sigmoid(v):
    return np.array([sigmoid_function(value) for value in v])