
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import math
from pgmpy.readwrite import BIFReader
from pgmpy.sampling.Sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator,BDeu, HillClimbSearch
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

def explog(logr,datatest,bics,sizes,loglis):
    logr.fits()
    logli0 = logr.scorell(datatest)
    bic0 = logr.akaike(logr.model)
    size0 = size(logr.model)
    bics.append(bic0)
    sizes.append(size0)
    loglis.append(logli0)


def exptree(dummy,v,datatest,bics,sizes,loglis,s=10):

    tree2 = probabilitytree()
    tree2.fit(dummy.dummycases,dummy.fvars,v, names = dummy.na,s=s)
    logli4 = tree2.valuate(dummy.transform(datatest),s=s)
    size4 = tree2.size()
    logli4s = tree2.valuate(dummy.dummycases,s=s)
    bic4 =  logli4s*dummy.dummycases.shape[0] - size4
    bics.append(bic4)
    sizes.append(size4)
    loglis.append(logli4)
    

        


def sizet(t):
  return reduce(mul,t.cardinality[1:])*(t.cardinality[0]-1)




def sizet2(t):
    return reduce(mul,t.shape[1:],1)*(t.shape[0]-1)

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

def countingnon(input,output):
  filei = open(input,'r')
  fileo = open(output,"w")
  line = filei.readline()
  lines = filei.readlines()
  for line in lines:
   
      line = line.strip()
      reader = BIFReader("./Networks/"+line)
      print(line)
      network = reader.get_model()
      total = 0
      extr = 0
      for v in network.nodes():
          par = network.get_parents(v) 
          # print(v,par) 
          table = network.get_cpds(node=v).values

          y = sizet2(table) 
         
          if len(par)>1 and y>64:
              x = np.count_nonzero(table==0.0)
              total += y
              extr += x
      if total>0:
        fileo.write(line + " , "+ str(total)+ "," + str(extr)+ "," + str(extr/total)+ "," + str(1-extr/total)+"\n")
          

  
  
 
def experiment(input,output):
  
  K=11
  filei = open(input,'r')
  fileo = open(output,"w")
  line = filei.readline()
  sizesa = list(map(int, line.split()))

  
  lines = filei.readlines()
  
  # dls = dict()
  # dss = dict()

  sal = "net,size,logtable,sizetable,logtree,sizetree,loglrdummy,sizelrdummy,logtreedummy,sizetreedummy,  loglgpair,sizelgpair,   logtreepair,sizetreepair,loglgld,sizelgld,  logtreeld,sizetreeld,loglgldpair,sizeldlgpair,logtreeldpair,sizetreeldpair\n"

  

  fileo.write(sal)


  
  for info in lines:
      (line,reps) = info.split()
      rep = int(reps)
      reader = BIFReader("./Networks/"+line)
      print(line)
      

      network = reader.get_model()
      

        
      sampler = BayesianModelSampling(network)
      datatest = sampler.forward_sample(size=10000)
      # datatestn = convert(datatest,network.states)
      transformcat(datatest, network.states)

      

      for x in sizesa:
        lls = []
        ss = []
        
 

        for j in range(rep):


         database = sampler.forward_sample(size=x)
         transformcat(database, network.states)

         for v in network.nodes():
          par = network.get_parents(v) 
          values = len(pd.unique(database[v]))
          print(v,par) 

          print(v,pd.unique(database[v]),network.states[v])
          print("PADRES")
          remo = []
          for  w in par:
            print(w,pd.unique(database[w]),network.states[w])
            if len(pd.unique(database[w])) == 1:
              remo.append(w)

          if remo:
            par = [x for x in par if x not in remo]           

           
          if len(par)>1 and values == network.get_cardinality(v):
             bics = []
             sizes = []
             loglis = []
             table = tfit(database,par,  v, network.states,s=15,weighted=False)
             size0 = sizet(table)
             print(size0)
             if size0<=64:
                 print("Small size")
                 continue
             logli0 = valuate(table,datatest)
             logli0s = valuate(table,database)
             bic0 = logli0s*database.shape[0] - size0             
             bics.append(bic0)
             sizes.append(size0)
             loglis.append(logli0)


             tree = probabilitytree()
             tree.fit(database,par,v, names = network.states,s=15)
             logli3 = tree.valuate(datatest,s=15)
             logli3s = tree.valuate(database,s=15)
             size3 = tree.size()
             bic3 = logli3s*database.shape[0] - size3
             bics.append(bic3)
             sizes.append(size3)
             loglis.append(logli3)

             logr = generalizedlr(v,par,database,l=0.5)
             explog(logr,datatest,bics,sizes,loglis)
             dummy = logr.dummycases
             dummy2 = dummy.copy()
             exptree(dummy,v,datatest,bics,sizes,loglis,s=15)

             dummy.expandpair()
             explog(logr,datatest,bics,sizes,loglis)
             exptree(dummy,v,datatest,bics,sizes,loglis,s=15)

             dummy2.expandldad()

             logr.dummycases = dummy2
             dummy = dummy2
             explog(logr,datatest,bics,sizes,loglis)
             exptree(dummy,v,datatest,bics,sizes,loglis,s=15)
          
             dummy.expandpair()
             explog(logr,datatest,bics,sizes,loglis)
             exptree(dummy,v,datatest,bics,sizes,loglis,s=15)

             sal = line+ ',' + str(x) 

             
             for i in range(len(loglis)):
                 sal = sal+','+ str(loglis[i])+ ',' + str(sizes[i]) 
            
             sal = sal +"\n"
             
             
             fileo.write(sal)

             

             

      


             
             
            #  lls.append(loglis)
            #  ss.append(sizes)
                
        # for i in range(K):
        #     lili = [z[i] for z in lls]
        #     sizei = [z[i] for z in ss]
        #     dls[(line,x,i)] = lili 
        #     dss[(line,x,i)] = sizei 
        #     if len(lili)>0:
        #         averl =np.average(np.array(lili))
        #         avers = np.average(np.array(sizei))
        #         print(averl,avers)
        #         fileo.write("$" +str(i) + "," + str(averl) + ","+str(avers) +"\n")
        # print("\n")
  # print("TOTALES")
  # totaldl = dict()
  # totalsl = dict()
  # for x in sizesa:
  #     for i in range(K):
  #       totaldl[(x,i)] = []
  #       totalsl[(x,i)] = []
  # for line in lines:
  #       line = line.strip()
  #       for x in sizesa:
  #           print("Network: ", line, "Size: ", x)
  #           fileo.write("Network: " +  line +  "Size: " +  str(x)+"\n")
  #           for i in range(K):
  #                   totaldl[(x,i)] = totaldl[(x,i)] + dls[(line,x,i)]
  #                   totalsl[(x,i)] = totalsl[(x,i)] + dss[(line,x,i)]
  #                   if len(dls[(line,x,i)]) >0:
  #                       avera = np.average(np.array(dls[(line,x,i)]))
  #                       averso = np.average(np.array(dss[(line,x,i)]))
  #                       print(avera,averso)
  #                       fileo.write(str(avera) + "," +str(averso)+"\n" )
  
  # for x in sizes:
  #     print("Size: ", x)
  #     fileo.write("Size: " +  str(x))
  #     for i in range(K):
  #       avera = np.average(np.array(totaldl[(x,i)]))
  #       averso = np.average(np.array(totalsl[(x,i)]))
  #       print(avera,averso)
  #       fileo.write(str(avera) + "," +str(averso) )
  fileo.close()
          
   
             
               
           


 
 
def experimentprev(input,output):
  
  
  filei = open(input,'r')
  fileos = open('outs2',"w")
  fileol = open('outl2',"w")
  line = filei.readline()
  sizesa = list(map(int, line.split()))

  
  lines = filei.readlines()
  
  dls = dict()
  dss = dict()

  dls3 = dict()
  dss3 = dict()

  sal = 'net, size'
  for s in [1,2,5,10,15]:
          sal = sal+"," + 'logtable' + str(s) + "," + 'sizetable' + str(s) + ',' + 'logtree'+ str(s) + ',' + 'sizetree'+ str(s)
  sal = sal+'\n'

  fileos.write(sal)


  sal = 'net, size'
  for l in [0.01,0.05,0.1,0.5,1,2,4]:
          sal = sal+"," + 'loglr' + str(l) + "," + 'sizelr' + str(l) 
  sal = sal+'\n'

  fileol.write(sal)
  for info in lines:

      (line,reps) = info.split()
      rep = int(reps)
      reader = BIFReader("./Networks/"+line)  
      print(line)
      
      


      network = reader.get_model()
      

        
      sampler = BayesianModelSampling(network)
      datatest = sampler.forward_sample(size=1000)
      # datatestn = convert(datatest,network.states)
      transformcat(datatest, network.states)

      sizesa = [500]

      for x in sizesa:

       for j in range(rep):
        lls = []
        ss = []
        lls3 = []
        ss3 = []
        
        database = sampler.forward_sample(size=x)
        transformcat(database, network.states)



        for v in network.nodes():
          par = network.get_parents(v) 
          values = len(pd.unique(database[v]))

          print(v,pd.unique(database[v]),network.states[v])
          remo = []
          for  w in par:
            print(w,pd.unique(database[w]),network.states[w])
            if len(pd.unique(database[w])) == 1:
              remo.append(w)

          if remo:
            par = [x for x in par if x not in remo]           

           
          if len(par)>1 and values == network.get_cardinality(v):
             sizes = []
             loglis = []
             sizes3 = []
             loglis3 = []
             bics = []
             table = tfit(database,par,  v, network.states,s=s,weighted=False)
             size0 = sizet(table)
             print(size0)
             if size0<=64:
                 print("Small size")
                 continue
             

             for s in [1,2,5,10,15]:
                table = tfit(database,par,  v, network.states,s=s,weighted=False)
                size0 = sizet(table)
                logli0 = valuate(table,datatest)
                sizes.append(size0)
                loglis.append(logli0)
                

             for s in [1,2,5,10,15]:
                tree = probabilitytree()
                tree.fit(database,par,v, names = network.states,s=s)
                logli3 = tree.valuate(datatest,s=s)
                size3 = tree.size()
                sizes3.append(size3)
                loglis3.append(logli3)
             
             sal = line+ ',' + str(x) 
             


             for i in range(len(loglis)):
                 sal = sal+','+ str(loglis[i])+ ',' + str(sizes[i]) + ","  + str(loglis3[i])+ ',' + str(sizes3[i]) 
            
             sal = sal +"\n"
             
             
             fileos.write(sal)

             
             sizes = []
             loglis = []
             bics = []
             
             for l in [0.01,0.05,0.1,0.5,1,2,4]:                
              logr = generalizedlr(v,par,database,l=l)
              explog(logr,datatest,bics,sizes,loglis)               
               
                

            
             

             

      



             
             sal = line+ ',' + str(x) 
             
             for i in range(len(loglis)):
                 sal = sal+','+ str(loglis[i])+ ',' + str(sizes[i]) 
            
             sal = sal +"\n"
             
             
             fileol.write(sal)
            
   
  fileos.close()
  fileol.close()
          
   
             
        

 
 
def experimentlambda(input,output):
  
  K=5
  filei = open(input,'r')
  fileo = open(output,"w")
  line = filei.readline()
  sizesa = list(map(int, line.split()))

  
  lines = filei.readlines()
  
  sizesa = [1000]

  
  for info in lines:

      (line,reps) = info.split()
      rep = int(reps)
      reader = BIFReader("./Networks/"+line)
      print(line)
      
      sal = 'net, size'
      for l in [0.01,0.05,0.1,0.5,1,2,4]:
          sal = "," + 'loglr' + str(l) + "," + 'sizelr' + str(l) 
      sal = sal+'\n'


      network = reader.get_model()
      

        
      sampler = BayesianModelSampling(network)
      datatest = sampler.forward_sample(size=10000)
      # datatestn = convert(datatest,network.states)
      transformcat(datatest, network.states)

      

      for x in sizesa:

       for j in range(rep):
        
        
        database = sampler.forward_sample(size=x)
        transformcat(database, network.states)

        counter = 1

        for v in network.nodes():
          par = network.get_parents(v) 
          values = len(pd.unique(database[v]))

          print(v,pd.unique(database[v]),network.states[v])
          remo = []
          for  w in par:
            print(w,pd.unique(database[w]),network.states[w])
            if len(pd.unique(database[w])) == 1:
              remo.append(w)

          if remo:
            par = [x for x in par if x not in remo]           

           
          if len(par)>1 and values == network.get_cardinality(v):
             sizes = []
             loglis = []
             
             bics = []
             table = tfit(database,par,  v, network.states,s=s,weighted=False)
             size0 = sizet(table)
             print(size0)
             if size0<=64:
                 print("Small size")
                 continue
             counter +=1
             if counter<=165:
                continue

             for l in [0.01,0.05,0.1,0.5,1,2,4]:                
              logr = generalizedlr(v,par,database)
              explog(logr,datatest,bics,sizes,loglis,l=l)               
               
                

            
             

             

      



             
             sal = line+ ',' + str(x) 
             
             for i in range(len(loglis)):
                 sal = sal+','+ str(loglis[i])+ ',' + str(sizes[i]) 
            
             sal = sal +"\n"
             
             
             fileo.write(sal)
             
             
                
   
  fileo.close()
          
   
 
def counttables(input,output):
  
  K=5
  filei = open(input,'r')
  fileo = open(output,"w")
  line = filei.readline()
  sizesa = list(map(int, line.split()))

  
  lines = filei.readlines()
  
  dls = dict()
  dss = dict()

  dls3 = dict()
  dss3 = dict()

  
  for line in lines:
      line = line.strip()
      (line,rep) = line.split()
      reader = BIFReader("./Networks/"+line)
      print(line)
      
      network = reader.get_model()
      
      fileo.write(line)

        
      sampler = BayesianModelSampling(network)
      
      
      for size in sizesa:
        count = 0
        count2 = 0
        for i in range(10):
          database = sampler.forward_sample(size=size)
          # datatestn = convert(datatest,network.states)

          

          
            
          transformcat(database, network.states)



          for v in network.nodes():
              par = network.get_parents(v) 
              values = len(pd.unique(database[v]))

              remo = []
              for  w in par:
                print(w,pd.unique(database[w]),network.states[w])
                if len(pd.unique(database[w])) == 1:
                  remo.append(w)

              if remo:
                par = [x for x in par if x not in remo]           


              if len(par)>1:
                
                table = tfit(database,par,  v, network.states,s=2,weighted=False)
                size0 = sizet(table)
                print(size0)
                if size0<=64:
                    print("Small size")
                else:
                    count+=1
                    if not values == network.get_cardinality(v):
                      count2+=1
        fileo.write(','+ str(count2/count))
      fileo.write('\n')     


             
             
             
             
                




# experimentprev('input','salida')

experiment('input7.txt','outexpg7')

# counttables('input','outputtables')
