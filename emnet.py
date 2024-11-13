import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from pgmpy.readwrite import BIFReader
from dummyvar import *



logv = np.vectorize(math.log)


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



def predict_logref_model(model, data, target):
  instances = zip(data, target)
  probs = []
  
  l = list(model.classes_)
  for (features, label) in instances:
    
    prediction = model.predict_proba([features])
    
    # prediction is a list with a list inside
    probs.append(prediction[0][l.index(label)])

  # return probs
  return np.array(probs)

def weighted_value_counts(y, weights):
  counters = dict()

  # gets the different values for y
  labels = np.unique(y)
  

  # for each label add an initial counter equals to 0
  for label in labels:
    counters[label] = 0

   
  # now considers each sample
  for sample in range(y.shape[0]):
    counters[y[sample]] = counters[y[sample]]+weights[sample]

  # return counters as a no array
  return counters

def generate_simple_model(dataset, var, weights, s = 1):
  # select features and target
  target = select_target(dataset,var)
  # gets the counters for labels
  counts = weighted_value_counts(target.values, weights)



  counts.update((x,y+s ) for x, y in counts.items())

  # gets probs
  total =  sum(counts.values())
  counts.update((x,y/total) for x, y in counts.items())
  
  # makes predictions with this model
  w = predict_simple_model(counts, target.values)

  # return model (probs) and predictions
  return counts, w

def predict_simple_model(model,  target):
  probs = []
 
  for label in target:
    probs.append(model[label])

  # return probs
  return np.array(probs)

def generate_logreg_model(dataset, t,f,weights):

    # select features and target
    features = select_features(dataset,f)
    target = select_target(dataset,t)

    # learn logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='saga',penalty='l1', max_iter = 200)
    model.fit(features.values, target.values, sample_weight = weights)

    # predicts with log-linear model
    w = predict_logref_model(model, features.values, target.values)

    # return the model and the weights
    return model, w

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


def m_step(dummy,data,net,probs,pc,  s=2):
  # get target column
  

  counts = probs.sum(axis=0)
  counts = counts + s
  counts = counts/counts.sum()

  expect = probs*pc

  w2 = expect.sum(axis=1)

  w1 = 1-w2


  weights = np.ones((data.shape[0],len(pc)))
  ln = list(net.nodes())
  
  
  llgr = dict()
  lsimp = dict()
  

  # learn logistic regression model
  for var in net.nodes():
    if net.get_parents(var):
        lgr, wr = generate_logreg_model(dummy[var].dummycases, var,dummy[var].fvars, w1)
    else:
        lgr, wr = generate_simple_model(data, var, w1, s)
    
       
    llgr[var] = lgr


  # learn the simple models counting labels
    simp, ws = generate_simple_model(data, var, w1, s)
    lsimp[var] = simp
    for i in range(len(pc)):
       weights[:,i] = weights[:,i] *( pc[i] * ws + (1-pc[i]) * wr)    


  # return models and new weights
  return weights,llgr,lsimp,counts,w2

class emnet:

    def __init__(self,attr,net,changes,pchanges,type = 1):
        self.data = attr
        self.type = type
        self.models = dict()
        self.simple = dict()
        self.net = net
        self.changes = changes
        self.pchanges = pchanges
        self.dummy = dict()
        self.counts = pchanges
        self.computedummy()


        
    def computedummy(self):
        for var in self.net.nodes():
            if self.net.get_parents(var):
                self.dummy[var] = dummyvar(var,self.net.get_parents(var),self.data)
            

  

    def fit(self,eps = 0.5,s=2):
        
        weights = np.ones((self.data.shape[0],len(self.changes)))
        ln = list(self.net.nodes())
  
  
     
        lwr = []
        lws = []

        w = np.ones(self.data.shape[0])
        for var in self.net.nodes():
            if self.net.get_parents(var):
                lgr, wr = generate_logreg_model(self.dummy[var].dummycases, var,self.dummy[var].fvars, w)
            else:
                lgr, wr = generate_simple_model(self.data, var, w)
    
       
            self.models[var] = lgr


            simp, ws = generate_simple_model(self.data, var, w)
            self.simple[var] = simp
            for i in range(len(self.pchanges)):
                weights[:,i] = weights[:,i] *( self.changes[i] * ws + (1-self.changes[i]) * wr)    

  # initializes the value of best models

        probs = weights*self.changes

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
        while True:
            # perform expectation step
            probs, loglike = e_step(weights,self.changes,counts)
            print(loglike)
            if loglike > loglike_best+eps:
                loglike_best = loglike
                print("    improvement: " , loglike_best)
            
            else:
                break

            # perform maximization step
            weights,llgr,lsimp, counts, w2 = m_step(self.dummy,self.data,self.net,probs, self.changes, s)
            
        

            # makes a new expectation step for updating loglike
            # wr, ws, loglike = e_step(wr, ws, alpha_n)
            print("loglike: ", loglike)

            # checks for the improvement
            

        # return best models
        self.counts = counts
        self.models = llgr
        self.simple = lsimp
        return llgr, lsimp, counts, w2
                
            
    def predict(self,newdata):
            
        pco = self.counts.copy()
        weights = np.ones((newdata.shape[0],len(self.counts)))
        wei =  np.ones((newdata.shape[0],len(self.counts)))
        predictsimp = dict()
        predictlgr = dict()
        predictchange = dict()
        for var in self.net.nodes():

            target = select_target(newdata,var)
            if self.net.get_parents(var):
                features = select_features(newdata,self.net.get_parents(var))
                tdata = self.dummy[var].transform(newdata)
              
                
                predictlgr[var] =  predict_logref_model(self.models[var], tdata[self.dummy[var].fvars].values,target.values)
            else:
                predictlgr[var] = predict_simple_model(self.models[var] , target.values)

            predictsimp[var] = predict_simple_model(self.simple[var] , target.values)
            for i in range(len(self.counts)):
                weights[:,i] = weights[:,i] *( self.changes[i] * predictsimp[var] + (1-self.changes[i]) *predictlgr[var] ) 
    
        weights = pco*weights

        for var in self.net.nodes():
            for i in range(len(self.counts)):
                wei[:,i] = weights[:,i] /( self.changes[i] * predictsimp[var] + (1-self.changes[i]) *predictlgr[var] )

            chav = wei*self.changes
            
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
        exp2 = pco*self.changes
        exp = exp2.sum(axis = 1)

        return exp, predictchange


       



        
