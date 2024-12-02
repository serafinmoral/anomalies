# from https://www.kaggle.com/code/charel/learn-by-example-expectation-maximization/notebook

import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from pgmpy.readwrite import BIFReader
import pandas as pd
import random as rd
import networkx as nx
from emnet import *
import time


def transform(data,net):
    snames = net.states
    for var in net.nodes():
      data[var] = data[var].apply(lambda x: snames[var].index(x))

def transformcategor(data,net):
    for v in net.nodes():
        data[v] = data[v].astype('category')
    

def forward_sample_noisy(
        network,
        size=1,
        changes = (0,0.2,0.5,0.7,1),
        pchanges = (0.8,0.03,0.02,0.05,0.1),
        seed=None
    ):
        """
        Generates sample(s) from joint distribution of the bayesian network with noise


        Parameters
        ----------
        size: int
            size of sample to be generated

        

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: boolean
            Whether to show a progress bar of samples getting generated.

        

        Returns
        -------
        sampled: pandas.DataFrame
            The generated samples

        
        """
        sampled = pd.DataFrame(columns=list(network.nodes()))

        
        pbar = list(nx.topological_sort(network))

        if seed is not None:
            np.random.seed(seed)

        pchan = rd.choices(changes,pchanges,k=size)
        for j in range(size):
            row = dict()
            for node in pbar:
                    cha = rd.choices([True,False],(pchan[j],1-pchan[j]))
                    if cha[0]:
                       row[node] = rd.choice(network.states[node])
                    else:
                        cpd = network.get_cpds(node)
                    
                        states = network.states[node]



                        evidence = cpd.variables[:0:-1]
                    
                        if evidence:
                            evidence_values = [row[i] for i in evidence]
                            table = network.get_cpds(node)
                            slice_ = [slice(None)] * len(cpd.variables)
                            for var  in evidence:
                                var_index = cpd.variables.index(var)
                                slice_[var_index] = cpd.get_state_no(var,row[var])
                            weights = cpd.values[tuple(slice_)]

                           
                            
                            
                            
                        
                        else:
                            weights = cpd.values
                        row[node] = rd.choices(network.states[node],weights)[0]

            sampled.loc[len(sampled)] = row

        sampledn = sampled.copy()
        transformcategor(sampledn,network)

        # samples_df = _return_samples(sampled, network.state_names_map)

        return (sampled,sampledn,pchan)




reader = BIFReader("Networks/barley.bif")
net = reader.get_model()

(dfo,df,pchan) = forward_sample_noisy(net,size=5000)





# call the iteration method

changes = (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
pchanges = (0.9,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.055)

em = emnet(df,net,changes,pchanges)

llgr,lsimp,counts,w2  = em.fit()



st = dict()
c = dict()
for x in pchan:
   st[x] = 0
   c[x] = 0

for i in range(len(w2)):
   print(pchan[i], w2[i])
   c[pchan[i]] +=1
   st[pchan[i]] += w2[i]

for x in c:
   print(x)
   st[x] = st[x]/c[x]

print(st)

for i in range(len(changes)):
   print(changes[i], counts[i])

time.sleep(80)
(dfo2,df2,pchan2) = forward_sample_noisy(net,size=500)

poc,pvec = em.predict(df2)

for i in range (len(pchan2)):
   print("\n",pchan2[i], poc[i],"\n")
#    for var in pvec:
#       print(var,pvec[var][i])





