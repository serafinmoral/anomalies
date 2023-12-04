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


def transform(data,net):
    snames = net.states
    for var in net.nodes():
      data[var] = data[var].apply(lambda x: snames[var].index(x))

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
        transform(sampledn,network)

        # samples_df = _return_samples(sampled, network.state_names_map)

        return (sampled,sampledn,pchan)



# selection of columns with features
# @param object of dataframe class (pandas)
# @return a new data frame with first n-1 columns
def select_features(dataframe,f) :
    # determine the number of columns

    # selects all but the last
    features = dataframe[f]


    # return the selected columns
    return features

# select the last columns of the dataframe
# @param object of dataframe class (pandas)
# @return a new data with the last column
def select_target(dataframe,t) :
    # determine the number of columns
    target=  dataframe[t]


    # selects only the last columns

    # return the selected column
    return target

# convert a sklearn bunch after loading a dataset into
# dataset object of Bunch class (sklearn) to transform to
# a pandas dataframe
# @return dataframe (pandas)
def from_bunch_to_dataframe(dataset) :
  # determine the number of features
  n_features = len(dataset.feature_names)

  # makes logical vectors for column selections
  features_select = [True] * n_features + [False]
  target_select = [False] * n_features + [True]

  # makes a Bunch with the resul
  base = dataset.frame.loc[:, features_select]
  target = dataset.frame.loc[:, target_select]

  # join all of them into a single data frame
  base['Target'] = target

  # return base
  return base

# generates a count of values for the target variable but
# considering the weight of each sample
# @param y object of Series class with labels of classes
# @return np.arrray with weighted counters for each label
def weighted_value_counts(y, weights):
  counters = {}

  # gets the different values for y
  labels = np.unique(y)

  # for each label add an initial counter equals to 0
  for label in labels:
    counters[label] = 0

  # now considers each sample
  for sample in y:
    counters[sample] = counters[sample]+weights[sample]

  # return counters as a no array
  return np.array([*counters.values()])

# generates a logistic regression model from data and weights
# @param dataset object of dataframe class (pandas)
# @param weights vector of weight for samples
# @ return model and vector of predictions
def generate_logreg_model(dataset, t,f,weights):

  # select features and target
  features = select_features(dataset,f)
  target = select_target(dataset,t)

  # learn logistic regression model
  model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter = 200)
  model.fit(features.values, target.values, sample_weight = weights)

  # predicts with log-linear model
  w = predict_logref_model(model, features.values, target.values)

  # return the model and the weights
  return model, w

# generates a simple model counting the number of instances
# with each label and considering the corresponding weights
# @param dataset object of Bunch class
# @param weights vector of weight for samples
# @param s value to use for Laplace correction
# @ return model and vector of predictions
def generate_simple_model(dataset, var, weights, s = 1):
  # select features and target
  target = select_target(dataset,var)
  # gets the counters for labels
  counts = weighted_value_counts(target.values, weights)

  # add laplace correction
  counts = counts + s

  # gets probs
  probs = counts / (sum(counts))

  # makes predictions with this model
  w = predict_simple_model(probs, target.values)

  # return model (probs) and predictions
  return probs, w

# predict data using logreg model
# @param model of log linear regression
# @param data object of class DataFrame (pandas) with features
# @param target object of Series (pandas) with labels
# @param return np.array with
def predict_logref_model(model, data, target):
  instances = zip(data, target)
  probs = []
  for (features, label) in instances:
    prediction = model.predict_proba([features])

    # prediction is a list with a list inside
    probs.append(prediction[0][label])

  # return probs
  return np.array(probs)

# makes the prediction using the simple model
# @param model of log linear regression
# @param data object of class DataFrame (pandas) with features
# @param target object of Series (pandas) with labels
# @param return np.array with
def predict_simple_model(model,  target):
  probs = []
  for label in target:
    probs.append(model[label])

  # return probs
  return np.array(probs)

# performs the expectation step of the algorithm
# @param w1 vector of weights computed during expectation step
#        obtained with the first model
# @param w2 vector of weights computed during expectation step
#        obtained with the second model
# @param alpha parameter setting the weights of models
# @return new values of weights produced by both models and loglike
def e_step(weights, pc,w):
  probs = weights*w

  # computes denominator for normalization
  den = probs.sum(axis=1)

  size = den.shape[0]

  den = den.reshape((size,1))

  # normalize
  probs = probs/den
  




  # computes loglike
  logop = lambda t: math.log(t)
  loglike = sum(np.array([logop(xi) for xi in den]))

  # return weights and loglike
  return probs, loglike

# perform maximization step of the algorithm
# @param dataset object of class dataframe (pandas)
# @param w1 vector of weights computed during expectation step
#        obtained with the first model
# @param w2 vector of weights computed during expectation step
#        obtained with the second model
# @param s value for Laplace correction
# @param regression model, predictions, simple model, predictions
#        and new value of alpha
def m_step(dataset,net,probs,pc,  s=2):
  # get target column
  

  counts = probs.sum(axis=0)
  counts = counts + s
  counts = counts/counts.sum()

  expect = probs*pc

  w2 = expect.sum(axis=1)

  w1 = 1-w2


  weights = np.ones((dataset.shape[0],len(pc)))
  ln = list(net.nodes())
  
  
  llgr = dict()
  lsimp = dict()
  

  # learn logistic regression model
  for var in net.nodes():
    if net.get_parents(var):
        lgr, wr = generate_logreg_model(dataset, var,net.get_parents(var), w1)
    else:
        lgr, wr = generate_simple_model(dataset, var, w1, s)
    
       
    llgr[var] = lgr


  # learn the simple models counting labels
    simp, ws = generate_simple_model(dataset, var, w1, s)
    lsimp[var] = simp
    for i in range(len(pc)):
       weights[:,i] = weights[:,i] *( pc[i] * ws + (1-pc[i]) * wr)    


  # return models and new weights
  return weights,llgr,lsimp,counts,w2

def predict(net,df,llgr,lsimp,counts,changes):
   
   pco = counts.copy()
   weights = np.ones((df.shape[0],len(counts)))
   wei =  np.ones((df.shape[0],len(counts)))
   predictsimp = dict()
   predictlgr = dict()
   predictchange = dict()
   for var in net.nodes():

        target = select_target(df,var)
        if net.get_parents(var):
            features = select_features(df,net.get_parents(var))
            predictlgr[var] =  predict_logref_model(llgr[var], features.values,target.values)
        else:
            predictlgr[var] = predict_simple_model(llgr[var] , target.values)

        predictsimp[var] = predict_simple_model(lsimp[var] , target.values)
        for i in range(len(counts)):
           weights[:,i] = weights[:,i] *( changes[i] * predictsimp[var] + (1-changes[i]) *predictlgr[var] ) 
   
   weights = pco*weights

   for var in net.nodes():
        for i in range(len(counts)):
           wei[:,i] = weights[:,i] /( changes[i] * predictsimp[var] + (1-changes[i]) *predictlgr[var] )

        chav = wei*changes
        
        nochav = wei-chav
        ec = chav.sum(axis=1)
        nec = nochav.sum(axis=1)
        ec = ec*predictsimp[var]
        nec = nec* predictlgr[var]
        sc = ec+nec
        ec = ec/sc
        predictchange[var] = ec
        

        
   pco = weights
   den = pco.sum(axis=1)
   den = den.reshape(den.shape[0],1)
   pco = pco/den
   exp2 = pco*changes
   exp = exp2.sum(axis = 1)

   return exp, predictchange



    



       

     

# function implementing expectation-maximization algorithm
# @param iteration
# @param dataset object of class dataframe (pandas)
# @param alpha parameter defining the mix: alpha f1 + (1 - alpha) f2
# @param s value for Laplace correction
# @return best models found and last value of alpha
def em_algorithm(net, dataset,  pc,wc, s=1, epsilon=0.1, iterations = 30):
  # initialy all the instances have the same weight
  weights = np.ones((dataset.shape[0],len(pc)))
  ln = list(net.nodes())
  
  
  llgr = []
  lsimp = []
  lwr = []
  lws = []

  # learn logistic regression model
  w = np.ones(dataset.shape[0])
  for var in net.nodes():
    if net.get_parents(var):
        lgr, wr = generate_logreg_model(dataset, var,net.get_parents(var), w)
    else:
        lgr, wr = generate_simple_model(dataset, var, w, s)
    
       
    llgr.append(lgr)


  # learn the simple models counting labels
    simp, ws = generate_simple_model(dataset, var, w, s)
    lsimp.append(simp)
    for i in range(len(pc)):
       weights[:,i] = weights[:,i] *( pc[i] * ws + (1-pc[i]) * wr)    

  # initializes the value of best models

  probs = weights*wc

  # computes denominator for normalization
  den = probs.sum(axis=1)

  size = den.shape[0]

  den = den.reshape((size,1))

  # normalize
  probs = probs/den

  # initializes the value of loglike
  loglike_best = float('-inf')

  counts = probs.sum(axis=0)
  counts = counts + s
  counts = counts/counts.sum()

  # initializes alpha_n with alpha

  # loop of optimization
  for i in range(1, iterations+1):
    # perform expectation step
    probs, loglike = e_step(weights,pc,counts)
    print(loglike)
    if loglike > loglike_best+epsilon:
      loglike_best = loglike
      print("    improvement: " , loglike_best)
    
    else:
      break

    # perform maximization step
    weights,llgr,lsimp, counts, w2 = m_step(dataset,net,probs, pc, s)
    
   

    # makes a new expectation step for updating loglike
    # wr, ws, loglike = e_step(wr, ws, alpha_n)
    print("loglike: ", loglike)

    # checks for the improvement
    

  # return best models
  return llgr, lsimp, counts, w2






reader = BIFReader("barley.bif")
net = reader.get_model()

(dfo,df,pchan) = forward_sample_noisy(net,size=500)



# call the iteration method

changes = (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
pchanges = (0.9,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,    0.055)
llgr,lsimp,counts,w2  = em_algorithm(net,df, changes, pchanges, 30)



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


(dfo2,df2,pchan2) = forward_sample_noisy(net,size=500)

poc,pvec = predict(net,df2,llgr,lsimp,counts,changes)

for i in range (len(pchan2)):
   print("\n",pchan2[i], poc[i],"\n")
   for var in pvec:
      print(var,pvec[var][i])





