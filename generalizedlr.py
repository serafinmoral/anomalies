from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

import pandas as pd
import time
import math
import numpy as np
import itertools

def rep(vr):
    seq = [x.split('_')[0] for x in vr]
    seen = []
    unique_list = [x for x in seq if x not in seen and not seen.append(x)]
    return (len(unique_list)< len(vr))


class generalizedlr:


    def __init__(self, v, par, data):
        self.var = v
        self.parents = par
        self.dataset = data
        self.fvars = []
        self.operations = []
        self.dummycases = None
        self.model = None
        self.nv = 0
        self.prepare()


    def prepare(self):
        self.dummycases = pd.get_dummies(self.dataset, columns= self.parents, drop_first=True)
        for v in self.parents:
            cas = self.dataset[v].dtype.categories
            for i in range(1,len(cas)):
                nc = cas[i]
                self.fvars.append(v+'_'+cas[i])
                self.operations.append((1,v,cas[i]))
        self.nv = len(self.fvars)

    def tdummies(self,data):
        result = pd.get_dummies(data, columns= self.parents, drop_first=True)
        for x in self.operations:
            if x[0]==2:
                vr = x[1]
                name = vr[0]
                for i in range(1,len(vr)):
                    name = name + "-" + vr[i]
                cas = result[vr]
                array = cas.to_numpy()
                mult = array.prod(axis=1)         
                result[name] = mult
        return result
              



    def expand(self,k=2):
        s = set(range(self.nv))
        for h in itertools.combinations(s,k):
            vr = [self.fvars[i] for i in h]
            if rep(vr):
                continue
            name = vr[0]
            for i in range(1,len(vr)):
                name = name + "-" + vr[i]
            cas = self.dummycases[vr]
            array = cas.to_numpy()
            mult = array.prod(axis=1)
            if mult.sum() >= 5:
                self.dummycases[name] = mult
                self.fvars.append(name)
                self.operations.append((2,vr))



            
    def scorell(self,datatest):
        testd = self.tdummies(datatest)
        probs = self.model.predict_proba(testd[self.fvars])

        cat = list(self.model.classes_)
        print(cat)

        ind = datatest.apply(lambda x: cat.index(x[self.var]),axis= 1).to_numpy()
        
        lpro = []
        n = len(ind)
        for i in range(n):
            lpro.append(math.log(probs[i][ind[i]]))

        return np.average(np.array(lpro))

        






    def fit(self):
        model = LogisticRegression(multi_class='auto', solver='liblinear', max_iter = 200, penalty='l1')
        # model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter = 200, penalty='l2')

        

        model.fit(self.dummycases[self.fvars] , self.dataset[self.var])
        # print(model.coef_)
        x = self.bic(model)
     

        best = x

        KM = len(self.parents)

        k = 2
        while k<= KM:
            print(k)
            self.expand(k)
            oldvars = self.fvars.copy()
            model.fit(self.dummycases[self.fvars] , self.dataset[self.var])
            print(model.coef_)

            x = self.bic(model)
            if (x> best):
                best = x
                k+=1
            else:
                self.vars = oldvars
                model.fit(self.dummycases[self.fvars] , self.dataset[self.var])
                break
                
        self.model = model

        
    def simplify(self):
        model = LogisticRegression(multi_class='auto', solver='liblinear', max_iter = 200, penalty='l1')
        model.fit(self.dummycases[self.fvars] , self.dataset[self.var])
        coe = model.coef_
        order = abs(coe).sum(axis=0)
        h=sorted(zip(self.fvars,order), key=lambda x: x[1])
        best = self.akaike(model)
        print(best)
        for x in h:
            if len(self.fvars)<=2:
                break
            var = x[0]
            self.fvars.remove(var)
            print(len(self.fvars))
            model.fit(self.dummycases[self.fvars] , self.dataset[self.var])
            score = self.akaike(model)
            print(score)
            if score>best:
                print("variable " + var + " eliminada")
                best = score
            else:
                self.fvars.append(var)
                print("variable " + var + " NO eliminada")
        self.model = LogisticRegression(multi_class='auto', solver='liblinear', max_iter = 200, penalty='l1')
        self.model.fit(self.dummycases[self.fvars] , self.dataset[self.var]) 


    def bic(self,model):

        probs = model.predict_proba(self.dummycases[self.fvars])

        cat = list(model.classes_)

        ind = self.dataset.apply(lambda x: cat.index(x[self.var]),axis= 1).to_numpy()
        
        lpro = []
        n = len(ind)
        for i in range(n):
            lpro.append(math.log(probs[i][ind[i]]))

        bic = np.array(lpro).sum() - 0.5*math.log(n)*(np.count_nonzero(model.coef_)  + 1)
        

        

        return bic


    def akaike(self,model):

        probs = model.predict_proba(self.dummycases[self.fvars])

        cat = list(model.classes_)

        ind = self.dataset.apply(lambda x: cat.index(x[self.var]),axis= 1).to_numpy()
        
        lpro = []
        n = len(ind)
        for i in range(n):
            lpro.append(math.log(probs[i][ind[i]]))

        bic = np.array(lpro).sum() - (np.count_nonzero(model.coef_)  + 1)
        

        

        return bic





      
      
      
      



