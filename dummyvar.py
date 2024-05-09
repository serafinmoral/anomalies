
import pandas as pd
import numpy as np
from pgmpy.estimators import BDeuScore
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class dummyvar:

    def __init__(self,v,par,data):
        
        self.var = v
        self.parents = par
        self.dataset = data
        self.fvars = []
        self.operations = []
        self.frequencies = dict()
        self.dummycases = pd.get_dummies(self.dataset, columns= self.parents, drop_first=True)
        for x in self.parents:
            cas = self.dataset[x].dtype.categories
            for i in range(1,len(cas)):
                self.fvars.append(x+'_'+cas[i])
                self.operations.append((1,x,cas[i]))
        self.nv = len(self.fvars)
        self.na = dict()
        for x in self.fvars:
            self.na[x] = [0,1]
        self.na[self.var] = self.dataset[self.var].dtype.categories
        self.lda = None

    def transform(self,newdata):
        newcases = pd.get_dummies(newdata, columns = self.parents, drop_first=True)
        for x in self.operations:
            if x[0] == 2:
                (oper,k,i,j) = x
                newvar = 'OPER_'+str(k)+'_'+str(i)+'_'+str(j)
                v1 = self.fvars[i]
                v2 = self.fvars[j]
                if k==1:
                    newcases[newvar] = newcases[v1]*newcases[v2]
                elif k==2:
                    newcases[newvar] = newcases[v1]*(1-newcases[v2])
                elif k==3:
                    newcases[newvar] = (1-newcases[v1])*(newcases[v2])
                elif k==4:
                    newcases[newvar] = (1-newcases[v1])*(1-newcases[v2])
                elif k==5:
                    newcases[newvar] = (1-newcases[v1])*(1-newcases[v2])+ newcases[v1]*newcases[v2]
            elif x[0] == 3:
               (oper,clf,i) = x
               newvalues = clf.transform(newcases[self.fvars[:self.nv]])
               dnew = np.where(newvalues > 0, 1, 0)
               newvar = "LDA_" + str(i)
               newcases[newvar] = dnew[:,i]


           
           
        return newcases
    
    def expandlda(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.dummycases[self.fvars[:self.nv]], self.dummycases[self.var])
        self.lda = clf
        newvars = clf.transform(self.dummycases[self.fvars[:self.nv]])
        dnew = np.where(newvars > 0, 1, 0)
        (n,nvar)  = dnew.shape
        for i in range(nvar):
           newvar = "LDA_" + str(i)
           self.dummycases[newvar] = dnew[:,i]
           self.operations.append((3,clf,i))
           self.fvars.append(newvar)
           self.na[newvar] = [0,1]

           
        


    def expand(self,s=10):
        

        estimator = BDeuScore(data=self.dummycases, state_names=self.na, equivalent_sample_size=s)
        i = 0
        j = 1
        best = max([estimator.local_score(self.var,[v1]) for v1 in self.fvars])
        oper = [1,2,3,4,5]
        while i<len(self.fvars)-1:
            v1 = self.fvars[i]

            while j < len(self.fvars):
                v2 = self.fvars[j]   
                for k in oper:
                  newvar = 'OPER_'+str(k)+'_'+str(i)+'_'+str(j)
                  estimator.state_names[newvar] = [0,1]

                  if k==1:
                    self.dummycases[newvar] = self.dummycases[v1]*self.dummycases[v2]
                  elif k==2:
                    self.dummycases[newvar] = self.dummycases[v1]*(1-self.dummycases[v2])
                  elif k==3:
                    self.dummycases[newvar] = (1-self.dummycases[v1])*(self.dummycases[v2])
                  elif k==4:
                    self.dummycases[newvar] = (1-self.dummycases[v1])*(1-self.dummycases[v2])
                  elif k==5:
                    self.dummycases[newvar] = (1-self.dummycases[v1])*(1-self.dummycases[v2]) + self.dummycases[v1]*self.dummycases[v2]
                  

                  snew = estimator.local_score(self.var,[newvar])
                  if snew > best:
                    self.fvars.append(newvar)
                    self.na[newvar] = [0,1]
                    print("nueva variable",newvar,snew,best)
                    self.operations.append((2,k,i,j))
                    best = snew
                  else:
                    self.dummycases.drop(newvar, axis='columns')
                j+=1
            
            i+=1
            j = i+1


    



