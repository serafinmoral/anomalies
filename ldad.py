import pandas as pd
import numpy as np
from pgmpy.estimators import BicScore,BDeuScore
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools
pd.options.mode.chained_assignment = None 
from mdlp.discretization import MDLP
from pandas.api.types import CategoricalDtype
import warnings



warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def check(v,e):
    if str(v)==str(e):
        return 1
    else:
        return 0

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





def cnewvar(row,listp,listn):
   x = 1 if all(row[v] == 1 for v in listp) and all(row[v] == 0 for v in listn) else 0
   return x

# Class to define new variables from discrete variables 
# in a problem of computing a conditional probability table


class ldad:

    def __init__(self,v,attr,create=True):
        
        self.var = v
        self.attr = attr
        self.newdata = None
        self.fvars = []
        self.operations = []
        self.na = dict()
        self.nlda = 0
        self.cont = []
        self.disc = []
        self.dummyv = []



        for x in attr.columns:

            if  attr[x].dtypes == 'float64':
                self.cont.append(x)
            else:
                self.disc.append(x)
                self.attr[x] = self.attr[x].astype("category")

        
        if create:
            self.newdata = pd.get_dummies(self.attr, columns= self.disc, drop_first=True)
            for x in self.attr.columns:
                cas = self.attr[x].dtype.categories
                self.operations.append((0,x,cas[1:]))
                for i in range(1,len(cas)):
                    self.fvars.append(x+'_'+str(cas[i]))
                    self.dummyv.append(x+'_'+str(cas[i]))
            self.nv = len(self.fvars)
            
            for x in self.fvars:
                self.na[x] = [0,1]
            
            self.var = self.var.astype("category")
            for x in self.cont:
                self.newdata[x] = self.attr[x]
            
            self.na['class'] = self.var.dtype.categories

    def transform(self,data):
        result = pd.DataFrame(index = data.index)
        for op in self.operations:
            if op[0] == 0:
                (h,var,cases) = op
                for c in cases:
                   result[var+'_'+str(c)] =  data[var].apply(check,args= [c]) 
            elif op[0] == 1:
                (h,nld,vars,clf,transformer) = op
                newvars = clf.transform(result[vars])
                (n,nvar)  =newvars.shape
                discrete = transformer.transform(newvars)
                listc = ['LDA_'+str(nld)+'_'+str(i) for i in range(discrete.shape[1])]
                discretedf = pd.DataFrame(discrete, columns= listc,index=data.index)

                for i in range(discrete.shape[1]):
                    values = list(range(len(transformer.cut_points_[i])+1))
                    v='LDA_'+str(nld)+'_'+str(i)
                    discretedf[v].astype(CategoricalDtype(categories=values))
                    for x in values[1:]:
                        result[v+'_'+str(x)] =  discretedf[v].apply(check,args= [x])
                    



              
                
                
        return result       





    def findvars(self,cl):
        result = []
        for v in self.newdata.columns:
            h = v.split('_')
            if h[0] in cl:
                result.append(v)
        return result
        
    
    def expandldad(self,vars,K=20):

        clf = LinearDiscriminantAnalysis()
        clf.fit(self.newdata[vars],self.var)
        newvars = clf.transform(self.newdata[vars])
        (n,nvar)  =newvars.shape
 

        transformer = MDLP(min_split = 0.01)  

    
        transformer.fit(newvars,self.var.cat.codes)

        
        discrete = transformer.transform(newvars)

        
        nld = self.nlda
        self.nlda +=1

        listc = ['LDA_'+str(nld)+'_'+str(i) for i in range(discrete.shape[1])]


        discretedf = pd.DataFrame(discrete, columns= listc,index=self.newdata.index)
        self.operations.append((1,nld,vars,clf,transformer))

        for i in range(discrete.shape[1]):
            values = list(range(len(transformer.cut_points_[i])+1))
            v='LDA_'+str(nld)+'_'+str(i)
            self.na[v] = values
            discretedf[v].astype(CategoricalDtype(categories=values))
            for x in values[1:]:
                self.newdata[v+'_'+str(x)] =  discretedf[v].apply(check,args= [x])
                self.fvars.append(v+'_'+str(x))
       

        
                
 