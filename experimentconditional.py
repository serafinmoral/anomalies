
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import math
from pgmpy.readwrite import BIFReader
from pgmpy.sampling.Sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator,BDeuScore, HillClimbSearch
from pgmpy.estimators import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
import pandas as pd
import time
import random as rd
import networkx as nx
from functools import reduce
from operator import mul
from itertools import chain
from scipy.io import arff
from pandas.api.types import CategoricalDtype

from generalizedlr import *
from ProbabilityTree import *
from dummyvar import *

 
# Computes de size of a conditional probability distribution stored as a table (pgmpy)

def sizet(t):
  return reduce(mul,t.cardinality[1:])*(t.cardinality[0]-1)
  
def transformcat(data,cases):
   for v in data.columns:
      
      data[v] = data[v].astype(CategoricalDtype(categories=cases[v]))



    









def tfit(dataset,parents,node, names,s=2, weighted=False):
  node_cardinality = len(names[node])
  parents_cardinalities = [len(names[parent]) for parent in parents]
  cpd_shape = (node_cardinality, np.prod(parents_cardinalities, dtype=int))

  alpha = float(s) / (node_cardinality * np.prod(parents_cardinalities))
  pseudo_counts = np.ones(cpd_shape, dtype=float) * alpha
  if weighted and ("_weight" not in dataset.columns):
            raise ValueError("data must contain a `_weight` column if weighted=True")
  if not parents:
            # count how often each state of 'variable' occurred
            if weighted:
                state_count_data = dataset.groupby([node])["_weight"].sum()
            else:
                state_count_data = dataset.loc[:, node].value_counts()

            state_counts = (
                state_count_data.reindex(names[node])
                .fillna(0)
                .to_frame()
            )
  else:
    parents_states = [names[parent] for parent in parents]
    if weighted:
                state_count_data = (
                    dataset.groupby([node] + parents)["_weight"]
                    .sum()
                    .unstack(parents)
                )

    else:
                state_count_data = (
                    dataset.groupby([node] + parents, observed=True)
                    .size()
                    .unstack(parents)
                )

    if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays([state_count_data.columns])
    row_index = names[node]
    column_index = pd.MultiIndex.from_product(parents_states, names=parents)
    state_counts = state_count_data.reindex(
                    index=row_index, columns=column_index
                ).fillna(0)
  
  bayesian_counts = state_counts + pseudo_counts

  cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(bayesian_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: names[var] for var in chain([node], parents)},
        )
  cpd.normalize()
  return cpd


def valuate(table,dataset):
  factor = table.to_factor()
  s = dataset.shape[0]
  result = 0.0
  evidence = table.variables

  for line in dataset.iterrows():
    index = {v: line[1][v] for v in evidence}
    x = factor.get_value(**index)  
    result += math.log(x)
  return result/s


def getprobs(table,dataset):
  factor = table.to_factor()
  s = dataset.shape[0]
  result = []
  evidence = table.variables

  for line in dataset.iterrows():
    index = {v: line[1][v] for v in evidence}
    x = factor.get_value(**index)  
    result.append(x)
  return np.array(result)



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
  line = filei.readline()
  sizes = list(map(int, line.split()))

  
  lines = filei.readlines()
  
  dls = dict()
  dss = dict()

  
  for line in lines:
      line = line.strip()
      reader = BIFReader("./Networks/"+line)
      print(line)
      

      network = reader.get_model()
      

        
      sampler = BayesianModelSampling(network)
      datatest = sampler.forward_sample(size=10000)
      # datatestn = convert(datatest,network.states)
      

      for x in sizes:
        fileo.write("*"+line+","+str(x))
        lls = []
        ss = []
        
        database = sampler.forward_sample(size=x)
        transformcat(database, network.states)
        transformcat(datatest, network.states)




        for v in network.nodes():
          par = network.get_parents(v) 
          values = len(pd.unique(database[v]))

          
          if len(par)>2 and values == network.get_cardinality(v):
             print(v,par) 
             logr = generalizedlr(v,par,database)

             logr.fit()
             logli1 = logr.scorell(datatest)
             bic1 = logr.akaike(logr.model)
             size1 = size(logr.model)
             logr.simplify()
             logli2 = logr.scorell(datatest)
             size2 = size(logr.model)
             bic2 = logr.akaike(logr.model)

             


             table = tfit(database,par,  v, network.states,s=2,weighted=False)

             logli0 = valuate(table,datatest)
             logli0s = valuate(table,database)

             size0 = sizet(table)
             
             bic0 = logli0s*database.shape[0] - size0

             tree = probabilitytree()
             tree.fit(database,par,v, names = network.states,s=10)
             logli3 = tree.valuate(datatest)
             logli3s = tree.valuate(database)
             size3 = tree.size()


             

             bic3 = logli3s*database.shape[0] - size3

             dummy = dummyvar(v,par,database)
             tree2 = probabilitytree()
             na = dict()
             for x in dummy.fvars:
                na[x] = [0,1]
             na[v] = network.states[v]
             tree2.fit(dummy.dummycases,dummy.fvars,v, names = na,s=10)
             logli4 = tree2.valuate(dummy.transform(datatest))
             size4 = tree2.size()
             logli4s = tree2.valuate(dummy.dummycases)
             bic4 =  logli4s*database.shape[0] - size4
             time.sleep(10)
             dummy.expand()
             tree5 = probabilitytree()
             tree5.fit(dummy.dummycases,dummy.fvars,v, names = dummy.na,s=10)
             logli5 = tree5.valuate(dummy.transform(datatest))
             size5 = tree5.size()
             logli5s = tree5.valuate(dummy.dummycases)
             bic5 =  logli5s*database.shape[0] - size5
             dummy.expandlda()
             tree6 = probabilitytree()
             tree6.fit(dummy.dummycases,dummy.fvars,v, names = dummy.na,s=10)
             logli6 = tree6.valuate(dummy.transform(datatest))
             size6 = tree6.size()
             logli6s = tree6.valuate(dummy.dummycases)
             bic6 =  logli6s*database.shape[0] - size6

             

             if max([bic0,bic1,bic2,bic3,bic4,bic5,bic6]) == bic0:
                 loglib = logli0
                 sizeb = size0
             elif max([bic0,bic1,bic2,bic3,bic4,bic5,bic6]) == bic1: 
                 loglib = logli1
                 sizeb = size1 
             elif max([bic0,bic1,bic2,bic3,bic4,bic5,bic6]) == bic2: 
                 loglib = logli2
                 sizeb = size2
             elif max([bic0,bic1,bic2,bic3,bic4,bic5,bic6]) == bic3:
                loglib = logli3
                sizeb = size3
             elif max([bic0,bic1,bic2,bic3,bic4,bic5,bic6]) == bic4:
                loglib = logli4
                sizeb = size4
             elif max([bic0,bic1,bic2,bic3,bic4,bic5,bic6]) == bic5:
                loglib = logli5
                sizeb = size5
             else:
                loglib = logli6
                sizeb = size6
                 

             print(logli0,logli1,logli2, logli3, logli4, logli5, logli6, loglib)
             fileo.write(str(logli0)+","+str(logli1)+","+str(logli2)+","+ str(logli3)+","+ str(logli4)+","+str(loglib)+"\n")
             print(bic0,bic1,bic2,bic3,bic4,bic5,bic6)
             print(size0,size1,size2,size3,size4,size5,size6, sizeb)
             fileo.write(str(size0)+","+str(size1)+","+str(size2)+","+ str(size3)+","+str(sizeb)+"\n")
             lls.append((logli0,logli1,logli2,logli3,logli4,logli5,logli6,loglib))
             ss.append((size0,size1,size2,size3,size4,size5,size6,sizeb))
                
        for i in range(8):
            lili = [x[i] for x in lls]
            sizei = [x[i] for x in ss]
            dls[(line,x,i)] = lili 
            dss[(line,x,i)] = sizei 
            if len(lili)>0:
                averl =np.average(np.array(lili))
                avers = np.average(np.array(sizei))
                print(averl,avers)
                fileo.write("$" +str(i) + "," + str(averl) + ","+str(avers) )
        print("\n")
  print("TOTALES")
  totaldl = dict()
  totalsl = dict()
  for x in sizes:
      for i in range(8):
        totaldl[(x,i)] = []
        totalsl[(x,i)] = []
  for line in lines:
        line = line.strip()
        for x in sizes:
            print("Network: ", line, "Size: ", x)
            fileo.write("Network: " +  line +  "Size: " +  str(x))
            for i in range(6):
                    totaldl[(x,i)] = totaldl[(x,i)] + dls[(line,x,i)]
                    totalsl[(x,i)] = totalsl[(x,i)] + dss[(line,x,i)]
                    if len(dls[(line,x,i)]) >0:
                        avera = np.average(np.array(dls[(line,x,i)]))
                        averso = np.average(np.array(dss[(line,x,i)]))
                        print(avera,averso)
                        fileo.write(str(avera) + "," +str(averso) )
  
  for x in sizes:
      print("Size: ", x)
      fileo.write("Size: " +  str(x))
      for i in range(6):
        avera = np.average(np.array(totaldl[(x,i)]))
        averso = np.average(np.array(totalsl[(x,i)]))
        print(avera,averso)
        fileo.write(str(avera) + "," +str(averso) )
  fileo.close()
          
   
             
               
           



        






          

experiment('input','output2')
