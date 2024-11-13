from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from emclassifier import *


import numpy as np
import math
from pgmpy.sampling.Sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator,BDeuScore, HillClimbSearch
from pgmpy.estimators import TreeSearch
import pandas as pd
import time
import random as rd
import networkx as nx
from functools import reduce
from operator import mul
from itertools import chain
from itertools import combinations
from pgmpy.estimators import BicScore,BDeuScore


from scipy.io import arff
from pandas.api.types import CategoricalDtype

from generalizedlr import *
from ProbabilityTree import *
from ldad import *
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

 
# Computes de size of a conditional probability distribution stored as a table (pgmpy)

def explog(logr,datatest,bics,sizes,loglis):
    logr.fits()
    logli0 = logr.scorell(datatest)
    bic0 = logr.akaike(logr.model)
    size0 = size(logr.model)
    bics.append(bic0)
    sizes.append(size0)
    loglis.append(logli0)


def createnet(attr,tar):
   vars = list(attr.columns)
   v = tar.name
   result = nx.Graph()
   result.add_nodes_from(vars)

   complete = pd.concat([attr,tar],axis=1)

   for (v1,v2) in combinations(vars,2):
      bde = BDeuScore(data=complete,equivalent_sample_size=5)
      sc1 = bde.local_score(v1,[v])
      sc2 = bde.local_score(v1,[v,v2])
      if sc2> sc1:
        result.add_edge(v1,v2)
     
   return result
    



        


def sizet(t):
  return reduce(mul,t.cardinality[1:])*(t.cardinality[0]-1)
  

  


    









def getprobst(tree,dataset):
  s = dataset.shape[0]
  result = []

  for line in dataset.iterrows():
    x = tree.getprob(line[1],s=2)  
    result.append(x)
  return np.array(result)
 
def experiment(input,output):
  
  filei = open(input,'r')
  fileo = open(output,"w")


  
  lines = filei.readlines()

  for line in lines:
      line = line.strip()
      (name,code) = line.split(',')
      print(line)
      

      data = fetch_ucirepo(id=int(code))  

      df = data.data     

        

      vars = df.features
      tars = df.targets
      
    

      for v in tars.columns:
        tars[v] = tars[v].astype('category')
        trainX,testX,trainY,testY = train_test_split(vars,tars[v],test_size=0.20)


        labels = tars[v].dtype.categories    
        ld = ldad(trainY,trainX)
        clf1 = LogisticRegression(random_state=0,penalty='l1',solver='saga').fit(ld.newdata, trainY)
        testXm1 = ld.transform(testX)
        ac1 = clf1.score(testXm1,testY)
        print("Accuracy simple: ",ac1)
        y_prob = clf1.predict_proba(testXm1)
        ll1 = log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood simple: ",ll1)

        
        emclas = emclassifier(ld.newdata,trainY,clf1,labels)
        emclas.fit()
        normal,probe = emclas.probanormal(testXm1,testY)
        y_prob = emclas.predict_proba(testXm1)
        ll1em= log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood simple em: ",ll1em)

        ld.expandldad(ld.fvars.copy())

        clf2 = LogisticRegression(random_state=0,penalty='l1',solver='saga',max_iter = 300).fit(ld.newdata, trainY)
        testXm2 = ld.transform(testX)
        ac2 = clf2.score(testXm2,testY)
        print("Accuracy lda: ",ac2)
        
        
        ll2= log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood lda: ",ll2)
        

        net = createnet(trainX,trainY)
        cliques = nx.find_cliques(net)

        ld2 = ldad(trainY,trainX)

        for c in cliques:
           if len(c)>1:
                nvars = ld2.findvars(c)
                ld2.expandldad(nvars)
        clf3 = LogisticRegression(random_state=0,penalty='l1',solver='saga',max_iter=300).fit(ld2.newdata, trainY)
        testXm3 = ld2.transform(testX)
        ac3 = clf3.score(testXm3,testY)
        print("Accuracy lda local: ",ac3)
        y_prob = clf3.predict_proba(testXm3)
        ll3= log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood lda local: ",ll3)




        cliques = nx.find_cliques(net)
        for c in cliques:
           if len(c)>1:
                nvars = ld.findvars(c)
                ld.expandldad(nvars)
        clf4 = LogisticRegression(random_state=0,penalty='l1',solver='saga',max_iter=300).fit(ld.newdata, trainY)
        testXm4 = ld.transform(testX)
        ac4 = clf4.score(testXm4,testY)
        print("Accuracy lda local: ",ac4)
        y_prob = clf4.predict_proba(testXm4)
        ll4= log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood lda local +: ",ll4)

        clf = MLPClassifier(random_state=1, max_iter=300).fit(ld.newdata[ld.dummyv], trainY)
        ac5 = clf.score(testXm1[ld.dummyv],testY)
        print("Neural network acc·", ac5 )
        y_prob = clf.predict_proba(testXm1[ld.dummyv])
        ll5= log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood neural network: ",ll5)


        clf = tree.DecisionTreeClassifier().fit(ld.newdata[ld.dummyv], trainY)
        ac6 = clf.score(testXm1[ld.dummyv],testY)
        print("Classification tree", ac6 )
        y_prob = clf.predict_proba(testXm1[ld.dummyv])
        ll6= log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood classification tree: ",ll6)                


        clf = HistGradientBoostingClassifier().fit(ld.newdata[ld.dummyv], trainY)
        ac7 = clf.score(testXm1[ld.dummyv],testY)
        print("Boosting tree", ac7 )
        y_prob = clf.predict_proba(testXm1[ld.dummyv])
        ll7= log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood boosting tree: ",ll7)                



        clf = RandomForestClassifier().fit(ld.newdata[ld.dummyv], trainY)
        ac8 = clf.score(testXm1[ld.dummyv],testY)
        print("Classification random forest", ac8 )
        y_prob = clf.predict_proba(testXm1[ld.dummyv])
        ll8= log_loss(testY, y_prob,labels = labels) 
        print("Loglikelihood random forest: ",ll8)                


        
           
        



             

             
 

             



  



        






          

experiment('inputlda','outputlda')
