from sympy import sympify, Symbol
import numpy as np

def symbolize(s):
    s1=s.replace('.','*')
    s2=s1.replace('^','**')
    s3=sympify(s2)
    
    return(s3)

def eval_multinomial(s,vals=None,symbolic_eval=False):
    sym_s=symbolize(s)
    sym_set=sym_s.atoms(Symbol)
    sym_lst=[]
    for s in sym_set:
        sym_lst.append(str(s))
    sym_lst.sort()
    if symbolic_eval==False and len(sym_set)!=len(vals):
        print("Length of the input values did not match number of variables and symbolic evaluation is not selected")
        return None
    else:
        if type(vals)==list:
            sub=list(zip(sym_lst,vals))
        elif type(vals)==dict:
            l=list(vals.keys())
            l.sort()
            lst=[]
            for i in l:
                lst.append(vals[i])
            sub=list(zip(sym_lst,lst))
        elif type(vals)==tuple:
            sub=list(zip(sym_lst,list(vals)))
        result=sym_s.subs(sub)
    return result

def gen_classification_symbolic(m=None,n_samples=100):
    sym_m=sympify(m)
    n_features=len(sym_m.atoms(Symbol))
    evals_binary=[]
    lst_features=[]
    n=0
    for i in range(n_features):
        lst_features.append(np.random.uniform(-1.0, 1.0, n_samples))

    lst_features=np.array(lst_features)
    lst_features = np.round(lst_features, 1)
    lst_features=lst_features.T
    for i in range(n_samples):
        temp = eval_multinomial(m,vals=list(lst_features[i])).evalf(n=1)

        if temp > 0.0:
            evals_binary.append('Hyperbola')
        elif temp < 0.0:
            evals_binary.append('Ellipse')
        else:
            n = n+1
            evals_binary.append('Parabola')

    print ('Number of samples Parabola: ',n)
    evals_binary=np.array(evals_binary)
    evals_binary=evals_binary.reshape(n_samples,1)
    lst_features=lst_features.reshape(n_samples,n_features)
    x=np.hstack((lst_features,evals_binary))
    
    return x


