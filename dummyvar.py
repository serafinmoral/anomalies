
import pandas as pd
import numpy as np
from pgmpy.estimators import BDeuScore
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools
pd.options.mode.chained_assignment = None 
from mdlp.discretization import MDLP
from pandas.api.types import CategoricalDtype
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def red(lista,K=20):
    for i in range(len(lista)):
        L = len(lista[i])
        if L>K:
            print("changing")
            pos = np.linspace(0,L-1, K).astype(np.int32)
            lista[i] = np.array([lista[i][j] for j in pos])

def rep(vr):
    seq = [x.split('_')[0] for x in vr]
    seen = []
    unique_list = [x for x in seq if x not in seen and not seen.append(x)]
    return (len(unique_list)< len(vr))

def sameldavar(l1,l2):
    l1d = set()
    l2d = set()
    for x in l1:
        (y,z) = x[::-1].split('_',1)
        l1d.add(z)
    for x in l2:
        (y,z) = x[::-1].split('_',1)
        l2d.add(z)
    if l1d.intersection(l2d):
        return True
    else:
        return False



def cnewvar(row,listp,listn):
   x = 1 if all(row[v] == 1 for v in listp) and all(row[v] == 0 for v in listn) else 0
   return x

# Class to define new variables from discrete variables 
# in a problem of computing a conditional probability table


class dummyvar:

    def __init__(self,v,par,data,create=True):
        
        self.var = v
        self.parents = par
        self.dataset = data
        self.fvars = []
        self.operations = []
        self.frequencies = dict()
        self.lda = None
        self.transforma = None
        self.amp = None
        self.ldad = False
        self.na = dict()
        self.dummycases = None
        self.delvar = []
        if create:
            self.dummycases = pd.get_dummies(self.dataset, columns= self.parents, drop_first=True)
            for x in self.parents:
                cas = self.dataset[x].dtype.categories
                for i in range(1,len(cas)):
                    self.fvars.append(x+'_'+cas[i])
                    self.operations.append((0,x,cas[i],{x+'_'+cas[i]}))
            self.nv = len(self.fvars)
            
            for x in self.fvars:
                self.na[x] = [0,1]
            self.na[self.var] = self.dataset[self.var].dtype.categories

    def copy(self):
       newd = dummyvar(self.var,self.parents,self.dataset,create=False)
       newd.dummycases = self.dummycases.copy()
       newd.fvars = self.fvars.copy()
       newd.operations = self.operations.copy()
       newd.frequencies = self.frequencies.copy()
       if self.lda:
         newd.lda = self.lda
       newd.transforma = self.transforma
       newd.ldad = self.ldad
       newd.amp = self.amp
       newd.na = self.na.copy()
       newd.nv = self.nv
       newd.delvar = self.delvar.copy()
       return newd

    def restrict(self,node,value):
       newdummy = dummyvar(self.var,self.parents,self.dataset, create= False)
       newdummy.fvars = self.fvars
       newdummy.na = self.na
       newdummy.nv = self.nv
       newdummy.operations = self.operations
       newdummy.dummycases = self.dummycases.loc[self.dummycases[node] == value ]
       return newdummy



    def transform(self,newdata):
        newcases = pd.get_dummies(newdata, columns = self.parents, drop_first=True)
        if self.ldad:
            newvars = self.lda.transform(newcases[self.fvars[:self.nv]])
            normal = newvars/self.amp
            discrete = self.transforma.transform(normal)
            listc = ['LDA_'+str(i) for i in range(discrete.shape[1])]
            discretedf = pd.DataFrame(discrete, columns= listc)
            for v in discretedf.columns:
                print(v)
                print(self.na[v])
                discretedf[v] = discretedf[v].astype(CategoricalDtype(categories=self.na[v]))
            dummydf = pd.get_dummies(discretedf,columns=listc,drop_first=True)  
        
            newcases = pd.concat([newcases,dummydf],axis = 1)

        for x in self.operations:
            if x[0] == 2:
                (oper,k,i,j,l) = x
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
            elif x[0]==1:
                vr = x[1]
                name = vr[0]
                for i in range(1,len(vr)):
                    name = name + "-" + vr[i]
                cas = newcases[vr]
                array = cas.to_numpy()
                mult = array.prod(axis=1)         
                newcases[name] = mult
            elif x[0] == 3:
               (oper,clf,i,[l]) = x
               newvalues = clf.transform(newcases[self.fvars[:self.nv]])
               dnew = np.where(newvalues > 0, 1, 0)
               newvar = "LDA_" + str(i)
               newcases[newvar] = dnew[:,i]
            elif x[0]==4:
                (oper,listn,listp) = x
                newvar = ''
                for v in listn:
                    newvar = newvar + v + "_0"
                for v in listp:
                    newvar = newvar + v + "_1"
                newcases[newvar] = newcases.apply(lambda row: cnewvar(row,listp,listn),axis=1)
                for v in listn:
                    newcases[newvar] = newcases[newvar]*(1-newcases[v])
                for v in listp:
                    newcases[newvar] = newcases[newvar]*(newcases[v])
           
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
           self.operations.append((3,clf,i,{newvar}))
           self.fvars.append(newvar)
           self.na[newvar] = [0,1]

    def expandldad(self,K=20):

        clf = LinearDiscriminantAnalysis()
        clf.fit(self.dummycases[self.fvars[:self.nv]], self.dummycases[self.var])
        self.lda = clf
        newvars = clf.transform(self.dummycases[self.fvars[:self.nv]])
        (n,nvar)  =newvars.shape
        # amp = newvars.max(axis=0) - newvars.min(axis=0)
        # normal = newvars/amp
        normal = newvars
        amp = 1
        transformer = MDLP(min_split = 0.01)  

        self.amp =amp
        transformer.fit(normal,self.dummycases[self.var].cat.codes)

        red(transformer.cut_points_)
         
        self.transforma = transformer
        self.ldad = True  
        discrete = transformer.transform(normal)

        listc = ['LDA_'+str(i) for i in range(discrete.shape[1])]
        

        discretedf = pd.DataFrame(discrete, columns= listc)
        for i in range(discrete.shape[1]):
            values = list(range(len(transformer.cut_points_[i])+1))
            v='LDA_'+str(i)
            if len(values)>K:
                print("Mirar")
            print(v,values)
           
            self.na[v] = values
            discretedf[v].astype(CategoricalDtype(categories=values))

       
        dummydf = pd.get_dummies(discretedf,columns=listc,drop_first=True)  
        
        self.dummycases = pd.concat([self.dummycases,dummydf],axis = 1)
        self.fvars = self.fvars + list(dummydf.columns)
        for x in dummydf.columns:
            (n1,n2,n3) = x.split('_')
            self.operations.append((6,'LDAd',x,clf,transformer,{x}))
            self.na[x] = [0,1]

        
                
          

    
    def expandlr(self,k=2):
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
                self.operations.append((1,vr))

    def expandpairld(self,s=2):
        estimator = BDeuScore(data=self.dummycases, state_names=self.na, equivalent_sample_size=s)
        s0 = estimator.local_score(self.var,[])

        i = 0
        j = 1
        H = len(self.fvars)
        while i<H:
            v1 = self.fvars[i]
            if not v1.startswith('LDA'):
                i+=1
                continue
            s1 = estimator.local_score(self.var,[v1])
            if self.operations[i][0] in  [0,3]:
                (h1,h2,h3,l1) = self.operations[i]
            elif self.operations[i][0] == 6:
                (h1,h2,h3,h4,h5,l1) = self.operations[i]
            while j < H:
                v2 = self.fvars[j]   
                if not v2.startswith('LDA'):
                    j+=1
                    continue
                newvar = 'OPER_5_'+str(i)+'_'+str(j)
                estimator.state_names[newvar] = [0,1]
                if self.operations[j][0] in  [0,3]:
                    (h1,h2,h3,l2) = self.operations[j]
                elif self.operations[j][0] == 2:
                    (h1,h2,h3,h4,l2) = self.operations[j]
                elif self.operations[j][0] == 6:
                    (h1,h2,h3,h4,h5,l2) = self.operations[j]
                if  len(l1.union(l2))>=3:
                        j+=1
                        continue
                if sameldavar(l1,l2):
                    j+=1
                    continue
                    
                self.dummycases[newvar] = (1-self.dummycases[v1])*(1-self.dummycases[v2]) + self.dummycases[v1]*self.dummycases[v2]
                s2 = estimator.local_score(self.var,[v2])

                snew = estimator.local_score(self.var,[newvar])
                if snew > max(s1,s2)+0.02*abs(max(s1,s2)) :
                    self.fvars.append(newvar)
                    self.na[newvar] = [0,1]
                    print("nueva variables rl",newvar,snew,s0,s1,s2,l1.union(l2))
                    self.operations.append((2,5,i,j,l1.union(l2)))
                else:
                    self.dummycases.drop(newvar, axis='columns')
                j+=1
            
            i+=1
            j = i+1

        



    def expandpair(self,s=2):
        estimator = BDeuScore(data=self.dummycases, state_names=self.na, equivalent_sample_size=s)
        i = 0
        j = 1
        H = (self.nv)

        estimator = BDeuScore(data=self.dummycases, state_names=self.na, equivalent_sample_size=s)
        i = 0
        j = 1
        H = (self.nv)
        while i<H:
            v1 = self.fvars[i]
            s1 = estimator.local_score(self.var,[v1])
            if self.operations[i][0] in  [0,3]:
                (h1,h2,h3,l1) = self.operations[i]
            elif self.operations[i][0] == 6:
                (h1,h2,h3,h4,h5,l1) = self.operations[i]
            while j < self.nv:
                v2 = self.fvars[j]   
                newvar = 'OPER_5_'+str(i)+'_'+str(j)
                estimator.state_names[newvar] = [0,1]
                if self.operations[j][0] in  [0,3]:
                    (h1,h2,h3,l2) = self.operations[j]
                elif self.operations[j][0] == 2:
                    (h1,h2,h3,h4,l2) = self.operations[j]
                elif self.operations[j][0] == 6:
                    (h1,h2,h3,h4,h5,l2) = self.operations[j]
                if  len(l1.union(l2))>=3:
                     j+=1
                     continue
                if sameldavar(l1,l2):
                    j+=1
                    continue
                   
                self.dummycases[newvar] = (1-self.dummycases[v1])*(1-self.dummycases[v2]) + self.dummycases[v1]*self.dummycases[v2]
                s2 = estimator.local_score(self.var,[v2])

                snew = estimator.local_score(self.var,[newvar])
                if snew > max(s1,s2)+0.1*abs(max(s1,s2)):
                    self.fvars.append(newvar)
                    self.na[newvar] = [0,1]
                    print("nueva variable rl",newvar,snew,s1,s2,l1.union(l2))
                    self.operations.append((2,5,i,j,l1.union(l2)))
                else:
                    self.dummycases.drop(newvar, axis='columns')
                j+=1
            
            i+=1
            j = i+1

           
 
    def expand(self,s=2):
        

        estimator = BDeuScore(data=self.dummycases, state_names=self.na, equivalent_sample_size=s)
        i = 0
        j = 1
        best = max([estimator.local_score(self.var,[v1]) for v1 in self.fvars])
        oper = [5]
        H = len(self.fvars)
        while i<H:
            v1 = self.fvars[i]
            (q1,q2,q3,l1) = self.operations[i]
            while j < len(self.fvars):
                v2 = self.fvars[j]
                if self.operations[j][0] in  [0,3]:
                    (h1,h2,h3,l2) = self.operations[j]
                elif self.operations[j][0] == 2:
                    (h1,h2,h3,h4,l2) = self.operations[j]
                # if  len(l1.union(l2))>=3:
                #      j+=1
                #      break  
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
                    self.operations.append((2,k,i,j,l1.union(l2)))
                    best = snew
                  else:
                    self.dummycases.drop(newvar, axis='columns')
                j+=1
            
            i+=1
            j = i+1


    



