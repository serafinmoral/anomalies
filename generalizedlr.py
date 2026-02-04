from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import statsmodels.api as sm
from scipy.stats.distributions import chi2

import pandas as pd
import time
import math
import numpy as np
import itertools
from dummyvar import *


pd.options.mode.chained_assignment = None 


def size(model):
    return np.count_nonzero(model.coef_)  + 1

def rep(vr):
    seq = [x.split('_')[0] for x in vr]
    seen = []
    unique_list = [x for x in seq if x not in seen and not seen.append(x)]
    return (len(unique_list)< len(vr))

def createformula(target,attributes):
    all_columns = ' + '.join(attributes)
    formula = target + "~" + all_columns
    return formula 




class generalizedlr:


    def __init__(self, v, par, data,l=1):
        self.var = v
        self.parents = par
        self.dataset = data
        self.fvars = []
        self.operations = []
        self.dummycases = dummyvar(v,par,data)
        self.model = None
        self.nv = len(self.dummycases.fvars)
        self.l = l


    def prepare(self):
        self.dummycases = pd.get_dummies(self.dataset, columns= self.parents, drop_first=True)
        for v in self.parents:
            cas = self.dataset[v].dtype.categories
            for i in range(1,len(cas)):
                nc = cas[i]
                self.fvars.append(v+'_'+cas[i])
                self.operations.append((1,v,cas[i]))
        self.nv = len(self.fvars)

 
              

    def expand2(self):
       i = 0
       j = 1
       N = self.dummycases.shape[0]
       H = len(self.fvars)
       model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter = 200, penalty='none')

       while i<H:
            v1 = self.fvars[i]
            while j < len(self.fvars):
                v2 = self.fvars[j]
                newvar = 'OPER_5_'+str(i)+'_'+str(j)
                self.dummycases[newvar] = (1-self.dummycases[v1])*(1-self.dummycases[v2]) + self.dummycases[v1]*self.dummycases[v2]
                andcases = self.dummycases[v1] * self.dummycases[v2]
                if len(andcases.unique())>1:
                    model.fit(self.dummycases[[v1,v2]],self.dummycases[self.var])
                    ak1 = self.akaike(model,[v1,v2])
                    model.fit(self.dummycases[[v1,v2,newvar]],self.dummycases[self.var])
                    ak2 = self.akaike(model,[v1,v2,newvar])
                    if ak2>ak1:
                        print("nueva variable ", newvar,ak2-ak1)
                        self.fvars.append(newvar)
                        self.operations.append((3,5,i,j))
                    else:
                        self.dummycases.drop(newvar,axis=1)
                j+= 1
            i+=1
            j= i+1
                # if test < t:
                #     self.fvars.append(newvar)
                #     self.na[newvar] = [0,1]
                #     print("nueva variable lg",newvar,p11,pt11)
                #     self.operations.append((2,5,i,j))


            
    def scorell(self,datatest, method=1):
        testd = self.dummycases.transform(datatest)
        if method==1:
            if self.dummycases.delvar:
                vars = self.dummycases.fvars.copy()
                for x in self.dummycases.delvar:
                    vars.remove(x) 
            else:
                vars = self.dummycases.fvars
            probs = self.model.predict_proba(testd[vars])
            cat = list(self.model.classes_)

        elif method==2:
            probs = np.array(self.model.predict(exog=testd[self.dummycases.fvar]))
            cat = list(self.model.model._ynames_map.values())
        # print(cat)
        ind = datatest.apply(lambda x: cat.index(x[self.var]),axis= 1).to_numpy()
        # print(ind)
        lpro = []
        n = len(ind)
        for i in range(n):
            lpro.append(math.log(probs[i][ind[i]]))

        return np.average(np.array(lpro))

        




    def fit2(self):

        model = LogisticRegression(multi_class='auto', solver='liblinear', max_iter = 200, penalty='l1')
        # model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter = 200, penalty='l2')

        self.expand2()

        model.fit(self.dummycases[self.fvars] , self.dataset[self.var])
        # print(model.coef_)
     

        self.model = model


    def fit3(self):
        model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter = 200, penalty='none')
        model.fit(self.dummycases[self.fvars] , self.dataset[self.var])
        best = self.akaike(model)
        self.model = model
        i=0
        j=1
        while i<len(self.fvars):
            v1 = self.fvars[i]
            while j < len(self.fvars):
                v2 = self.fvars[j]
                newvar = 'OPER_5_'+str(i)+'_'+str(j)
                self.dummycases[newvar] = (1-self.dummycases[v1])*(1-self.dummycases[v2]) + self.dummycases[v1]*self.dummycases[v2]
                andcases = self.dummycases[v1] * self.dummycases[v2]
                if len(andcases.unique())>1:
                    self.fvars.append(newvar)
                    model.fit(self.dummycases[self.fvars],self.dummycases[self.var])
                    ak2 = self.akaike(model)
                    if ak2>best:
                        print("nueva variable ", newvar,ak2,best)
                        self.operations.append((3,5,i,j))
                        best = ak2

                    else:
                        self.dummycases.drop(newvar,axis=1)
                        self.fvars.pop()
                j+=1
            i+=1
            j=i+1
        model.fit(self.dummycases[self.fvars],self.dummycases[self.var])
        self.model = model
        return self.model
                    
                    

    def fitexpand(self):
        self.dummycases.expandpair()
        model = LogisticRegression(solver='saga', max_iter = 200, l1_ratio = 1.0,C=1/self.l)
        dummy = self.dummycases.dummycases

        model.fit(dummy[self.dummycases.fvars] , dummy[self.var])

        self.model = model

    def fits(self):
        model = LogisticRegression(solver='saga', max_iter = 200, l1_ratio = 1.0,C=1/self.l)
        # model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter = 200, penalty='l2')

        dummy = self.dummycases.dummycases

        model.fit(dummy[self.dummycases.fvars] , dummy[self.var])
        self.model = model


    def fit(self):
        model = LogisticRegression(solver='saga', max_iter = 200, l1_ratio = 1.0,C=1/self.l)

        dummy = self.dummycases.dummycases

        model.fit(dummy[self.dummycases.fvars] , dummy[self.var])
        # print(model.coef_)
        x = self.akaike(model,k=0.5)
     

        best = x

        KM = len(self.parents)
        # KM = 2
        k = 2
        while k<= KM:
            self.dummycases.expandlr(k)
            oldvars = self.fvars.copy()
            model.fit(dummy[self.dummycases.fvars] , dummy[self.var])

            x = self.akaike(model,k=0.5)
            k+=1
            if (x> best):
                best = x
                k+=1
            else:
                self.vars = oldvars
                model.fit(dummy[self.dummycases.fvars] , dummy[self.var])
                break
                
        self.model = model

    def simplify3(self):
        model = LogisticRegression(multi_class='auto', solver='liblinear', max_iter = 200, penalty='l1')
        model.fit(self.dummycases[self.fvars] , self.dataset[self.var])
        self.model = model


        
    def simplify(self):
        model = LogisticRegression(multi_class='auto', solver='liblinear', max_iter = 200, penalty='l1')
        # model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter = 200, penalty='l2')

        vars = self.dummycases.fvars.copy()

        model.fit(self.dummycases.dummycases[vars] , self.dataset[self.var])
        coe = model.coef_
        order = abs(coe).sum(axis=0)
        h=sorted(zip(self.fvars,order), key=lambda x: x[1])
        best = self.akaike(model,k=0.5)
        
        # print(best)
        for x in h:
            if len(self.fvars)<=2:
                break
            var = x[0]
            vars.remove(var)
            # print(len(self.fvars))
            model.fit(self.dummycases.dummycases[vars] , self.dataset[self.var])
            score = self.akaike(model,k=0.5)
            # print(score)
            if score>best:
                # print("variable " + var + " eliminada")
                best = score
                self.dummycases.delvar.append(var)

            else:
                vars.append(var)
        # self.model = LogisticRegression(multi_class='auto', solver='liblinear', max_iter = 200, penalty='l1')
        self.model.fit(self.dummycases.dummycases[vars] , self.dataset[self.var]) 


    def bic(self,model):

        probs = model.predict_proba(self.dummycases[self.fvars])

        cat = list(model.classes_)

        ind = self.dataset.apply(lambda x: cat.index(x[self.var]),axis= 1).to_numpy()
        
        lpro = []
        n = len(ind)
        for i in range(n):
            lpro.append(math.log(probs[i][ind[i]]))

        bic = np.array(lpro).sum() - 0.5*math.log(n)*(size(model))
        

        

        return bic
    



    def akaike(self,model,k=1,lista = []):


        if lista:
            probs = model.predict_proba(self.dummycases.dummycases[lista])
        else:
            probs = model.predict_proba(self.dummycases.dummycases[self.dummycases.fvars])



        cat = list(model.classes_)

        ind = self.dataset.apply(lambda x: cat.index(x[self.var]),axis= 1).to_numpy()
        
        lpro = []
        n = len(ind)
        for i in range(n):
            lpro.append(math.log(probs[i][ind[i]]))

        bic = np.array(lpro).sum() - k*(size(model))
        

        

        return bic





      
      
      
      




