# from https://www.kaggle.com/code/charel/learn-by-example-expectation-maximization/notebook
 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from pgmpy.readwrite import BIFReader
from pgmpy.sampling.Sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator,BDeuScore
import pandas as pd
import time
import random as rd
import networkx as nx

def e_stepn(weights, pc,w):
  probs = weights*w
  den = probs.sum(axis=1)
  size = den.shape[0]
  den = den.reshape((size,1))
  probs = probs/den
  logop = lambda t: math.log(t)
  loglike = sum(np.array([logop(xi) for xi in den]))

  return probs, loglike


def m_stepn(dataset,net,probs,pc,  s=2, version='ta', models = dict()):
  # get target column
  

  counts = probs.sum(axis=0)
  counts = counts + s
  counts = counts/counts.sum()

  expect = probs*pc

  w2 = expect.sum(axis=1)

  w1 = 1-w2

  nc = dataset.shape[0]
  weights = np.ones((nc,len(pc)))
  
  
  lwt = dict()
  lws = dict()

  # learn logistic regression model
  bayest = BayesianEstimator(model=net, data=dataset, state_names=net.states)

  dataset['_weight'] = w1
  

  # learn logistic regression model
  for v in net.nodes():
    if version=='ta':
      table = bayest.estimate_cpd(v, prior_type="BDeu", equivalent_sample_size=2,weighted=True)
      models[v] = table
      lwt =getprobs(table,dataset)
    elif version=='tr':
      models[v].updatew(dataset,  v, net.states, s=2)
      lwt = getprobst(models[v],dataset)
    lws = np.full(nc,1/len(net.states[v]))

    for i in range(len(pc)):
       weights[:,i] = weights[:,i] *( pc[i] * lws + (1-pc[i]) * lwt) 
  
  return weights,models,counts,w2


def em_algorithmn(net, dataset,  pc,wc, s=2, epsilon=0.1, iterations = 30, version='ta'):
  # initialy all the instances have the same weight
  nc = dataset.shape[0]
  weights = np.ones((nc,len(pc)))
  
  
  models = dict()
  lwt = dict()
  lws = dict()

  # learn logistic regression model
  w = np.ones(nc)
  bayest = BayesianEstimator(model=net, data=dataset, state_names=net.states)

  dataset['_weight'] = w
  for v in net.nodes():
    if version == 'ta':
      table = bayest.estimate_cpd(v, prior_type="BDeu", equivalent_sample_size=2)
      models[v] = table
      lwt[v] = getprobs(table,dataset)
    elif version=="tr":
      par = net.get_parents(v)
      tree = probabilitytree()
      tree.fit(dataset,par,v, names = net.states,s=10)
      lwt[v] = getprobst(tree,dataset)
      models[v] = tree

    lws[v] = np.full(nc,1/len(net.states[v]))

    for i in range(len(pc)):
       weights[:,i] = weights[:,i] *( pc[i] * lws[v] + (1-pc[i]) * lwt[v]) 
  
  probs = weights*wc
  den = probs.sum(axis=1)
  size = den.shape[0]
  den = den.reshape((size,1))
  probs = probs/den
  loglike_best = float('-inf')
  counts = probs.sum(axis=0)
  counts = counts + s
  counts = counts/counts.sum()

  for i in range(1, iterations+1):
    probs, loglike = e_stepn(weights,pc,counts)
    if loglike > loglike_best+epsilon:
      loglike_best = loglike
      print("Improvement: " , loglike_best)
    
    else:
      break

    # perform maximization step
    weights,models, counts, w2 = m_stepn(dataset,net,probs, pc, s, version=version, models=models)
    
   

    # makes a new expectation step for updating loglike
    # wr, ws, loglike = e_step(wr, ws, alpha_n)

    # checks for the improvement
    

  # return best models
  return models, counts, w2





def transformt(data,net):
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
        transformt(sampledn,network)

        # samples_df = _return_samples(sampled, network.state_names_map)

        return (sampled,sampledn,pchan)



def select(data,features,target,names,s, bas=[]):
  scores = dict()
  estimator = BDeuScore(data=data, state_names=names, equivalent_sample_size=s)
  for x in features:
    bas.append(x)
    scores[x] = estimator.local_score(target,bas) 
    bas.remove(x)

  basicsco = estimator.local_score(target,bas)
  
  v = max(scores,key=scores.get)

  return (v,scores[v]-basicsco)



  

def testindep(data,var,target,names,s):
  estimator = BDeuScore(data=data, state_names=names, equivalent_sample_size=s)

  score = estimator.local_score(target,[var]) 

  basicsco = estimator.local_score(target,[])
  
  return score-basicsco


  


class probabilitytree:

  def __init__(self):
    node = None
    frequencies = dict()
    chidren = None
    model = None
    leaf = True



  def fit(self,data, features , target, names, s, double=False):
    frequencies = data[target].value_counts().to_dict()
    if not features or all(x==0 for x in frequencies):
      self.frequencies = frequencies
      self.node = target
      self.leaf = True
      for x in names[target]:
        if not x in self.frequencies:
          self.frequencies[x] = 0
    else:
      (node,score) = select(data,features,target,names,s)
      if score <=0:
        if double and len(features)>1:
          features.remove(node)
      
          (node2,score2)= select(data,features,target,names,s, [node])
          features.append(node)
        if not double or len(features)<=1 or score2<=0:
            self.frequencies = frequencies
            self.node = target
            self.leaf = True
            for x in names[target]:
              if not x in self.frequencies:
                self.frequencies[x] = 0 
            return self
      
      self.leaf = False
      self.node = node
      self.children = dict()
      features.remove(node)
      for x in names[node]:
        ch = probabilitytree()
        datax = data.loc[data[node] == x]
        ch.fit(datax, features , target, names, s/len(names[node]))
        self.children[x] = ch
      features.append(node)

    return self
  
  def fittot(self,data, features , target, names, s):
      self.frequencies = data[target].value_counts()
      
      for x in names[target]:
          if not x in self.frequencies:
            self.frequencies[x] = 0


      if not features:
        self.node = target
        self.leaf = True

      else:
        tot = sum(x != 0 for x in self.frequencies)
        if tot <=0:
          (node,score) = select(data,features,target,names,s)
          print("solo uno", [x for x in self.frequencies], score)
          
          self.node = target
          self.leaf = True
          
        else:
          (node,score) = select(data,features,target,names,s)

          self.leaf = False
          self.node = node
          self.children = dict()
          self.frequencies = dict()
          features.remove(node)
          for x in names[node]:
            ch = probabilitytree()
            datax = data.loc[data[node] == x]
            ch.fittot(datax, features , target, names, s/len(names[node]))
            self.children[x] = ch
          features.append(node)
      
      return self


  def updatew(self,data, v, states, s=2):
    if not self.leaf:
      for x in self.children.keys():
        datax = data.loc[data[self.node] == x]
        self.children[x].updatew(datax,v,states,s)
    else:
        self.frequencies = data.groupby(v)['_weight'].sum().to_dict()
        
        
        for x in states[v]:
            if not x in self.frequencies:
              self.frequencies[x] = 0 



    
   

    return self

  def prune(self,data, target, names, s= 10):
    if self.leaf:
      return self
    else:
      for x in self.children:
        datax = data.loc[data[self.node] == x]
        self.children[x].prune(datax,target,names, s/len(names[self.node]) )
      
      if all(self.children[x].leaf for x in self.children):
        score = testindep(data,self.node,target,names,s)
        if score<=0:
          self.leaf = True
          self.node = target
          self.children = None
          self.frequencies = data[target].value_counts()
          for x in names[target]:
            if not x in self.frequencies:
              self.frequencies[x] = 0 

    return self


  
    
    
  def size(self):
    if self.leaf:
      return (len(self.frequencies)-1)
    else:
      return(sum([self.children[x].size() for x in self.children]))

  def getprob(self,values,s=2):
    v = self.node

    if not self.leaf:
      return self.children[values[v]].getprob(values,s/len(self.children))
    else:
      total = sum(self.frequencies.values())
      total += s
      t = self.frequencies[values[v]]+s/len(self.frequencies)
      return t/total
  
  def valuate(self,data):
    s = data.shape[0]
    result = 0.0

    for line in data.iterrows():
    
      x = self.getprob(line[1],s=2)  
      result += math.log(x)
    return result/s


  



# selection of columns with features
# @param object of dataframe class (pandas)
# @return a new data frame with first n-1 columns
def select_features(dataframe) :
    # determine the number of columns
    n_columns = dataframe.shape[1]

    # selects all but the last
    features = dataframe.iloc[:, 0:n_columns-1]

    # return the selected columns
    return features

# select the last columns of the dataframe
# @param object of dataframe class (pandas)
# @return a new data with the last column
def select_target(dataframe) :
    # determine the number of columns
    n_columns = dataframe.shape[1]

    # selects only the last columns
    target = dataframe.iloc[:, n_columns-1]

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
  # print("labels: ", labels)

  # for each label add an initial counter equals to 0
  for label in labels:
    counters[label] = 0

  # now considers each sample
  for sample in y:
    counters[sample] = counters[sample]+weights[sample]
  # print("counters: ", counters)

  # return counters as a no array
  return counters

# generates a logistic regression model from data and weights
# @param dataset object of dataframe class (pandas)
# @param weights vector of weight for samples
# @ return model and vector of predictions
def generate_logreg_model(dataset, weights):

  # select features and target
  features = select_features(dataset)
  target = select_target(dataset)

  # learn logistic regression model
  model = LogisticRegression(multi_class='auto', solver='newton-cg', max_iter = 200)
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
def generate_simple_model(dataset, weights, s = 1):
  # select features and target
  features = select_features(dataset)
  target = select_target(dataset)
  probs= {}
  # gets the counters for labels
  counts = weighted_value_counts(target.values, weights)

  # add laplace correction
  

  for x in counts:
    counts[x]+= s
  tot = 0.0
  for x in counts:
    tot += counts[x]
  for x in counts:
    probs[x] = counts[x]/tot
  # gets probs
  

  # makes predictions with this model
  w = predict_simple_model(probs, features.values, target.values)
  # print("simple weight: ", w)

  # return model (probs) and predictions
  return probs, w

# predict data using logreg model
# @param model of log linear regression
# @param data object of class DataFrame (pandas) with features
# @param target object of Series (pandas) with labels
# @param return np.array with
def predict_logref_model(model, data, target):
  instances = zip(data, target)
  lcla = list(model.classes_)

  probs = []
  for (features, label) in instances:
    prediction = model.predict_proba([features])
    try:
      i = lcla.index(label)
    except:
      i = -1
    # print(prediction[0])
    # prediction is a list with a list inside
    if i>= 0:
      probs.append(prediction[0][i])
    else:
      probs.append(0.00001)
  # return probs
  return np.array(probs)

def valuatelogreg(model,data,target):
  probs = predict_logref_model(model,data,target)
  x = map(math.log,probs)
  return sum(x)/len(probs)

def valuatelogregm(lgr,sim,alpha,data,target):
  probs = predict_logref_model(lgr,data,target)
  probs2 = predict_simple_model(sim,data,target)
  fprob = alpha*probs+(1-alpha)*probs2
  x = map(math.log,fprob)
  return sum(x)/len(probs)



# makes the prediction using the simple model
# @param model of log linear regression
# @param data object of class DataFrame (pandas) with features
# @param target object of Series (pandas) with labels
# @param return np.array with
def predict_simple_model(model, data, target):
  instances = zip(data, target)


  probs = []
  for (features, label) in instances:
      x = model.get(label,-0.00001)
     

      probs.append(x)
    

  # return probs
  return np.array(probs)

# performs the expectation step of the algorithm
# @param w1 vector of weights computed during expectation step
#        obtained with the first model
# @param w2 vector of weights computed during expectation step
#        obtained with the second model
# @param alpha parameter setting the weights of models
# @return new values of weights produced by both models and loglike
def e_step(w1, w2, alpha):
  wp1 = w1 * alpha
  wp2 = w2 * (1 - alpha)

  # computes denominator for normalization
  den = wp1 + wp2

  # normalize
  wp1 /= den
  wp2 /= den

  # computes loglike
  logop = lambda t: math.log(t)
  loglike = sum(np.array([logop(xi) for xi in den]))

  # return weights and loglike
  return wp1, wp2, loglike

# perform maximization step of the algorithm
# @param dataset object of class dataframe (pandas)
# @param w1 vector of weights computed during expectation step
#        obtained with the first model
# @param w2 vector of weights computed during expectation step
#        obtained with the second model
# @param s value for Laplace correction
# @param regression model, predictions, simple model, predictions
#        and new value of alpha
def m_step(dataset, w1, w2, s):
  # get target column
  target = select_target(dataset)

  # determine the new value of alpha
  alpha_n = sum(w1)/len(target.values)

  # learn a new log-linear model having w1 as weights for
  # the data
  lgr, wr = generate_logreg_model(dataset, w1)

  # learn a new simple model having w2 as weights for the
  # data
  simp, ws = generate_simple_model(dataset, w2, s)

  # return models and new weights
  return lgr, wr, simp, ws, alpha_n

# function implementing expectation-maximization algorithm
# @param iteration
# @param dataset object of class dataframe (pandas)
# @param alpha parameter defining the mix: alpha f1 + (1 - alpha) f2
# @param s value for Laplace correction
# @return best models found and last value of alpha
def em_algorithm(iterations, dataset, alpha, s, epsilon=0.1):
  # initialy all the instances have the same weight
  # print("dimensions: " , dataset.shape[0])
  weights = np.ones(dataset.shape[0])

  # learn logistic regression model
  lgr, wr = generate_logreg_model(dataset, weights)

  # learn the simple models counting labels
  simp, ws = generate_simple_model(dataset, weights, s)

  # initializes the value of best models
  lgr_best = lgr
  simp_best = simp

  # initializes the value of loglike
  loglike_best = float('-inf')

  # initializes alpha_n with alpha
  alpha_n = alpha

  # loop of optimization
  for i in range(1, iterations+1):
    # perform expectation step
    wr_n, ws_n, loglike = e_step(wr, ws, alpha_n)
    if loglike > loglike_best+epsilon:
      loglike_best = loglike
      # print("    improvement: " , loglike_best)
    
    else:
      break

    # perform maximization step
    lgr_n, wr, simp_n, ws, alpha_n = m_step(dataset, wr_n, ws_n, s)
    lgr = lgr_n
    simp = simp_n
    # print("new alpha: ", alpha_n)
    # print("lgr: ", lgr.coef_, " - ", lgr.intercept_)
    # print("simp. model: ", simp_n)

    # makes a new expectation step for updating loglike
    # wr, ws, loglike = e_step(wr, ws, alpha_n)
    # print("loglike: ", loglike)

    # checks for the improvement
    

  # return best models
  return lgr_best, simp_best, alpha_n, wr_n


# function implementing expectation-maximization algorithm
# @param iteration
# @param dataset object of class dataframe (pandas)
# @param alpha parameter defining the mix: alpha f1 + (1 - alpha) f2
# @param s value for Laplace correction
# @return best models found and last value of alpha
def em_algorithmdim(iterations, dataset, k=2,  s=1, epsilon=0.1):
  # initialy all the instances have the same weight
  # print("dimensions: " , dataset.shape[0])
  weights = np.ones(dataset.shape[0])

  # learn logistic regression model
  lgr, wr = generate_logreg_model(dataset, weights)

  # learn the simple models counting labels
  simp, ws = generate_simple_model(dataset, weights, s)

  # initializes the value of best models
  lgr_best = lgr
  simp_best = simp

  # initializes the value of loglike
  loglike_best = float('-inf')

  # initializes alpha_n with alpha
  alpha_n = alpha

  # loop of optimization
  for i in range(1, iterations+1):
    # perform expectation step
    wr_n, ws_n, loglike = e_step(wr, ws, alpha_n)
    if loglike > loglike_best+epsilon:
      loglike_best = loglike
      # print("    improvement: " , loglike_best)
    
    else:
      break

    # perform maximization step
    lgr_n, wr, simp_n, ws, alpha_n = m_step(dataset, wr_n, ws_n, s)
    lgr = lgr_n
    simp = simp_n
    # print("new alpha: ", alpha_n)
    # print("lgr: ", lgr.coef_, " - ", lgr.intercept_)
    # print("simp. model: ", simp_n)

    # makes a new expectation step for updating loglike
    # wr, ws, loglike = e_step(wr, ws, alpha_n)
    # print("loglike: ", loglike)

    # checks for the improvement
    

  # return best models
  return lgr_best, simp_best, alpha_n, wr_n

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
    
def convert(data,names):
  result = data.copy()
  for v in data.columns.values:
    result[v] = result[v].transform(lambda x: names[v].index(x))


  return result

def experiment(input, output):
  filei = open(input,'r')
  fileo = open(output,"w")
  line = filei.readline()
  sizes = list(map(int, line.split()))
  results = dict()
  asize = dict()
  database = dict()
      
  
  


  lines = filei.readlines()


  av0= 0.0
  n = 0
  av3 = 0.0
  w0 = 0
  w3 = 0
  avs0 = 0
  avs3 = 0
  av4 = 0.0
  avs4 = 0
  w34 = 0
  w43  = 0
  w35 = 0
  w53 = 0
  av5 = 0
  avs5 = 0
  av6 = 0
  avs6 = 0


  for line in lines:
    line = line.strip()
    reader = BIFReader("./Networks/"+line)
    print(line)
    network = reader.get_model()
    for v in network.nodes():
      par = network.get_parents(v)
      
    sampler = BayesianModelSampling(network)
    datatest = sampler.forward_sample(size=10000)
    # datatestn = convert(datatest,network.states)
    

    for x in sizes:
      database = sampler.forward_sample(size=x)
      # databasen = convert(database,network.states)



      for v in network.nodes():
        par = network.get_parents(v) 
        size0 = 1.0
        size1= 0.0
        for v2 in par:
            size0 *= network.get_cardinality(v2)
            size1+= 1
        size0 *= (network.get_cardinality(v)-1)
        size1 *= network.get_cardinality(v) -1
        
        if len(par)>1:
          bayest = BayesianEstimator(model=network, data=database, state_names=network.states)

          table = bayest.estimate_cpd(v, prior_type="BDeu", equivalent_sample_size=2)
          logli0 = valuate(table,datatest)

          # features = databasen[par]
          # target = databasen[v]




          # model = LogisticRegression(multi_class='auto', solver='newton-cg', max_iter = 200)
          
          # model.fit(features.values, target.values)





          # features2 = datatestn[par]
          # target2 = datatestn[v]

          # dt = pd.merge(features,target,left_index=True, right_index=True)
          # logli1 = valuatelogreg(model,features2.values,target2.values)
          # lgr, simp, alpha, wr = em_algorithm(30, dt, 0.5, 1)

          # logli2 = valuatelogregm(lgr,simp,alpha,features2.values,target2.values)

          size0 = 1.0
          # size1= 0.0
          for v2 in par:
            size0 *= network.get_cardinality(v2)
          #   size1+= 1
          size0 *= (network.get_cardinality(v)-1)
          # size1 *= network.get_cardinality(v) -1
          # logd0 = valuate(table,database)
          # logd1 = valuatelogreg(model,features.values,target.values)
          # bic0 = x*logd0-0.5*math.log(x)*size0
          # bic1 = x*logd1-0.5*math.log(x)*size1
          
          tree = probabilitytree()
          tree.fit(database,par,v, names = network.states,s=10)
          logli3 = tree.valuate(datatest)
          # print(size0,size1)
          avs0 += size0
          size3 = tree.size()
          avs3 += size3
          tree = probabilitytree()
          tree.fit(database,par,v, names = network.states, s=20)
          logli4 = tree.valuate(datatest)
          size4 = tree.size()
          tree = probabilitytree()
          tree.fit(database,par,v, names = network.states, s=10, double = True)
          logli5 = tree.valuate(datatest)
          size5 = tree.size()
          tree = probabilitytree()
          tree.fit(database,par,v, names = network.states, s=20, double = True)
          logli6 = tree.valuate(datatest)
          size6 = tree.size()

          n+=1
          av0 += logli0
          av3 += logli3
          av4 += logli4
          avs4 += size4
          av5 += logli5
          avs5 += size5
          av6 += logli6
          avs6 += size6


          if logli0>logli3+0.00001:
            w0+=1
          elif logli3>logli0+0.00001:
            w3+=1
          if logli3>logli4+0.00001:
            w34+=1
          elif logli4>logli3+0.00001:
            w43+=1
          if logli3>logli5+0.00001:
            w35+=1
          elif logli5>logli3+0.00001:
            w53+=1
          print(logli0, logli3, logli4,logli5, logli6,size0,size3,size4, size5, size6)
          print(av0/n,av3/n,av4/n,av5/n,av6/n)
          print(w0,w3)
          print(w34,w43)
          print(w35,w53)
          fileo.write(f'{line},{v},{size0},{x},{size3}, {size4},{size5},{size6},  {logli0}, {logli3}, {logli4},{logli5}, {logli6}\n')

          print(avs0/n,avs3/n,avs4/n,avs5/n,avs6/n)
      results[x,0] = av0/n
      results[x,3] = av3/n
      results[x,4] = av4/n
      results[x,5] = av5/n
      results[x,6] = av6/n

      asize[x,0] = avs0/n
      asize[x,3] = avs3/n
      asize[x,4] = avs4/n
      asize[x,5] = avs5/n
      asize[x,6] = avs6/n


  for x in sizes:
    print(x)
    print(results[x,0] ,results[x,3] ,results[x,4] ,results[x,5] ,results[x,6])
    print(asize[x,0], asize[x,3],asize[x,4],asize[x,5],asize[x,6] )

def transform(x):
  return {i: x[i] for i in x.index}

def transform2(x,v,par):

  h =  dict()
  h[v] = x[v]
  for i in par:
    h[i] = x[i]

  return h

def estimateprob(table,v,states,parvalues, eps = 0.001):
  p0 = 0.5
  n = len(states)
  phi= table.to_factor()
  tot = parvalues.shape[0]
     
  prob = np.array(parvalues.apply(lambda x: phi.get_value(**transform(x)),axis=1))

  proa = np.full(tot, 1/n)

  prop = (p0*proa)/(p0 * proa + (1-p0)*prob)
  p1 = prop.sum()/tot

  while(abs(p1-p0)>eps):
    print(p0)
    p0 = p1
    prop = (p0*proa)/(p0 * proa + (1-p0)*prob)
    p1 = prop.sum()/tot


  return(p1,prop)


def estimateprobem(bayest,v,par,states,parvalues, eps = 0.001):
  p0 = 0.5
  n = len(states)
  tot = parvalues.shape[0]

  table = bayest.estimate_cpd(v, prior_type="BDeu", equivalent_sample_size=2)

  prob = np.array(parvalues.apply(lambda x: table.to_factor().get_value(**transform2(x,v,par)),axis=1))

  proa = np.full(tot, 1/n)

  vlog = np.vectorize(math.log)
  like = vlog(p0 * proa + (1-p0)*prob)
  

  x = like.sum()

  old = 0

  first = True
  while x-old> eps or first:
    first = False
    prop = (p0*proa)/(p0 * proa + (1-p0)*prob)
    p1 = prop.sum()/tot
    parvalues['_weight'] = 1-prop

    table = bayest.estimate_cpd(v, prior_type="BDeu", equivalent_sample_size=2, weighted=True)
    old = x

    prob = np.array(parvalues.apply(lambda x: table.to_factor().get_value(**transform2(x,v,par)),axis=1))
    like = vlog(p1 * proa + (1-p1)*prob)
    x = like.sum()
    p0 = p1
    print(p0,x)


  return(p1,prop, table)



def estimateprobtem(bayest,v,par,states,parvalues, eps = 0.001):
  p0 = 0.5
  n = len(states)
  tot = parvalues.shape[0]

  tree = probabilitytree()
  tree.fit(parvalues,par,  v, states,s=2,double = True)
  prob = np.array(parvalues.apply(lambda x:tree.getprob(x),axis=1))


  proa = np.full(tot, 1/n)

  vlog = np.vectorize(math.log)
  like = vlog(p0 * proa + (1-p0)*prob)
  

  x = like.sum()

  old = 0

  first = True
  while x-old> eps or first:
    first = False
    prop = (p0*proa)/(p0 * proa + (1-p0)*prob)
    p1 = prop.sum()/tot
    parvalues['_weight'] = 1-prop
    tree.updatew(parvalues,  v, states, s=2)
    prob = np.array(parvalues.apply(lambda x:tree.getprob(x),axis=1))


    like = vlog(p1 * proa + (1-p1)*prob)

    old = x
    x = like.sum()
    p0 = p1


  return(p1,prop, tree)

def categorn(models, changes,pchanges, dataset, version='ta'):
  return 1

def categor(table,p0,v,states,parvalues, eps = 0.001):
  n = len(states)
  n = len(states)
  n = len(states)
  phi= table.to_factor()
  tot = parvalues.shape[0]
     
  prob = np.array(parvalues.apply(lambda x: phi.get_value(**transform(x)),axis=1))

  proa = np.full(tot, 1/n)

  prop = (p0*proa)/(p0 * proa + (1-p0)*prob)

  


  return prop



def categort(tree,p0,v,states,parvalues, eps = 0.001):
  n = len(states)
  tot = parvalues.shape[0]

  prob = np.array(parvalues.apply(lambda x:tree.getprob(x),axis=1))
  proa = np.full(tot, 1/n)

  prop = (p0*proa)/(p0 * proa + (1-p0)*prob)

  


  return prop


def estimateprobt(tree,v,states,parvalues, eps = 0.001):
  p0 = 0.5
  n = len(states)
  tot = parvalues.shape[0]
  prob = np.array(parvalues.apply(lambda x:tree.getprob(x),axis=1))

  proa = np.full(tot, 1/n)

  prop = (p0*proa)/(p0 * proa + (1-p0)*prob)
  p1 = prop.sum()/tot

  while(abs(p1-p0)>eps):
    print(p0)
    p0 = p1
    prop = (p0*proa)/(p0 * proa + (1-p0)*prob)
    p1 = prop.sum()/tot


  return(p1,prop)

def randommodify(vvalues,states,alpha=0.1):
    
    x = vvalues.values.shape[0]
    randomvalues = np.random.choice(states, x)

    proba = np.random.uniform(0, 1,x)

    change = proba<=alpha

    for i in range(len(change)):
            if change[i]:
              vvalues.loc[i] = randomvalues[i] 


def experiment4(input, output):
  filei = open(input,'r')
  fileo = open(output,"w")
  line = filei.readline()
  sizes = list(map(int, line.split()))
  lines = filei.readlines()

  for line in lines:
    line = line.strip()
    reader = BIFReader("./Networks/"+line)
    print(line)
    network = reader.get_model()
    for x in sizes:

      (dfo,df,pchan) = forward_sample_noisy(network,size=x)
      (dfot,dft,pchant) = forward_sample_noisy(network,size=1000)

      changes = (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
      pchanges = (0.9,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,    0.055)
      tables,counts,w2  = em_algorithmn(network,dfo, changes, pchanges, 30, version='tr')
      trees,countst,w2t  = em_algorithmn(network,dfo, changes, pchanges, 30, version='ta')


      st = dict()
      c = dict()
      stt = dict()
      for x in pchan:
        st[x] = 0
        c[x] = 0
  
        stt[x] = 0

      for i in range(len(w2)):
        print(pchan[i], w2[i],w2t[i])
        c[pchan[i]] +=1
        st[pchan[i]] += w2[i]
        stt[pchan[i]] += w2t[i]

      for x in c:
        print(x)
        st[x] = st[x]/c[x]
        stt[x] = stt[x]/c[x]

      print(st)
      print(stt)

      for i in range(len(changes)):
        print(changes[i], counts[i],countst[i])
      print(counts.sum(), countst.sum())




    





def experiment3(input, output,alpha=0.1):
  filei = open(input,'r')
  fileo = open(output,"w")
  line = filei.readline()
  sizes = list(map(int, line.split()))
  results = dict()
  asize = dict()
  database = dict()
      
  
  


  lines = filei.readlines()
  w = 0
  l = 0




  for line in lines:
    line = line.strip()
    reader = BIFReader("./Networks/"+line)
    print(line)
    network = reader.get_model()
    for v in network.nodes():
      par = network.get_parents(v)
      
    sampler = BayesianModelSampling(network)
    ts = 1000
    datatest = sampler.forward_sample(size=ts)


    for x in sizes:
      database = sampler.forward_sample(size=x)
      database2 = sampler.forward_sample(size=x)

      for v in network.nodes():
        par = network.get_parents(v) 
        size0 = 1.0
        size1= 0.0
        for v2 in par:
            size0 *= network.get_cardinality(v2)
            size1+= 1
        size0 *= (network.get_cardinality(v)-1)
        size1 *= network.get_cardinality(v) -1
        
        if len(par)>1:

          vvalues = database2[v].copy()
          vvalueso = vvalues.copy()

          randommodify(vvalues,network.states[v])
          database2[v] = vvalues
          
          database2['_weight'] = [1]*database2.shape[0]
          bayest = BayesianEstimator(model=network, data=database2, state_names=network.states)

          (p0,changee,table) = estimateprobem(bayest,v,par,network.states[v],database2)

          (p0t,changeet,tree) = estimateprobtem(bayest,v,par,network.states,database2)
          database2[v] = vvalueso



          

       

          time.sleep(3)
            
          

          


def experiment2(input, output,alpha=0.1):
  filei = open(input,'r')
  fileo = open(output,"w")
  line = filei.readline()
  sizes = list(map(int, line.split()))
  results = dict()
  asize = dict()
  database = dict()
      
  
  


  lines = filei.readlines()
  w = 0
  l = 0




  for line in lines:
    line = line.strip()
    reader = BIFReader("./Networks/"+line)
    print(line)
    network = reader.get_model()
    for v in network.nodes():
      par = network.get_parents(v)
      
    sampler = BayesianModelSampling(network)
    ts = 1000
    datatest = sampler.forward_sample(size=ts)


    for x in sizes:
      database = sampler.forward_sample(size=x)
      database2 =   sampler.forward_sample(size=x)
 
      for v in network.nodes():
        par = network.get_parents(v) 
        size0 = 1.0
        size1= 0.0
        for v2 in par:
            size0 *= network.get_cardinality(v2)
            size1+= 1
        size0 *= (network.get_cardinality(v)-1)
        size1 *= network.get_cardinality(v) -1
        
        if len(par)>1:
          bayest = BayesianEstimator(model=network, data=database, state_names=network.states)

          table = bayest.estimate_cpd(v, prior_type="BDeu", equivalent_sample_size=2)
          tree = probabilitytree()
          tree.fit(database,par,v, names = network.states,s=20, double=True)
          vvalues = database2[v].copy()
          parvalues = database2[par]
          vvaluest = datatest[v].copy()
          parvaluest = datatest[par]
          randomvalues = np.random.choice(network.states[v], x)
          randomvaluest = np.random.choice(network.states[v], ts)

          proba = np.random.uniform(0, 1,x)
          probat = np.random.uniform(0, 1,x)

          change = proba<=alpha
          changet = probat<=alpha

          for i in range(len(change)):
            if change[i]:
              vvalues.loc[i] = randomvalues[i] 
          for i in range(len(changet)):
            if changet[i]:
              vvaluest.loc[i] = randomvaluest[i] 
          parvalues[v] = vvalues
          parvaluest[v] = vvaluest

          (p0,changee) = estimateprob(table,v,network.states[v],parvalues)
          (p0t,changeet) = estimateprobt(tree,v,network.states[v],parvalues)


          sumc = 0.0
          sumnc  = 0.0
          nc = 0
          nnc = 0
          sl = 0.0
          slt = 0.0
          sumct = 0.0
          sumnct  = 0.0
          nct = 0
          nnct = 0
          for i in range(len(change)):
            if change[i]:
              sumc += changee[i]
              nc += 1
              sumct += changeet[i]
              nct += 1
              sl += math.log(changee[i])
              slt += math.log(changeet[i])

            else:
              sumnc += changee[i]
              nnc += 1
              sumnct += changeet[i]
              nnct += 1
              sl += math.log(1-changee[i])
              slt += math.log(1-changeet[i])

          print(p0,sumc/nc,sumnc/nnc)
          print(p0t,sumct/nct,sumnct/nnct)
          print(sl/x,slt/x, 1 if sl>slt else 0)

          changee2 = categor(table,p0,v,network.states[v],parvaluest)
          changeet2 = categort(tree,p0,v,network.states[v],parvaluest)
          sl2 = 0.0
          slt2 = 0.0
          sumc = 0.0
          sumnc  = 0.0
          nc = 0
          nnc = 0
          
          sumct = 0.0
          sumnct  = 0.0
          nct = 0
          nnct = 0

          
          for i in range(len(changet)):
            if changet[i]:
              sumc += changee2[i]
              nc += 1
              sumct += changeet2[i]
              nct += 1
              sl2 += math.log(changee2[i])
              slt2 += math.log(changeet2[i])
          else:
              sumnc += changee2[i]
              nnc += 1
              sumnct += changeet2[i]
              nnct += 1
              sl2 += math.log(1-changee2[i])
              slt2 += math.log(1-changeet2[i])
          if slt2 > sl2:
            w+=1
          elif slt2 < sl2:
            l+=1
          
          print(sl2/ts,slt2/ts , w, l)
          print(p0,sumc/nc,sumnc/nnc)
          print(p0t,sumct/nct,sumnct/nnct)

          time.sleep(3)
            
          

          


     


  for x in sizes:
    print(x)
    print(results[x,0] ,results[x,3] ,results[x,4] ,results[x,5] ,results[x,6])
    print(asize[x,0], asize[x,3],asize[x,4],asize[x,5],asize[x,6] )



   



    
      




experiment4('input','output')

# # read dataset
# # dataset = sklearn.datasets.fetch_covtype(as_frame = True)
# dataset = sklearn.datasets.load_iris(as_frame = True)
# # convert to data frame




# df = from_bunch_to_dataframe(dataset)
# random_vals = np.random.choice(df.Target.unique(), df.shape[0])
# proba = np.random.uniform(0, 100, df.shape[0])
# print(proba)
# h = df.Target.copy()
# df.Target = df.Target.where(proba > 20, random_vals)
# # df.Target = df.Target-1
# print("type of df: ", type(df))

# # call the iteration method
# lgr, simp, alpha, wr = em_algorithm(30, df, 0.5, 1)

# for i in range(len(wr)):
#   print(proba[i],wr[i], df.Target[i], h[i])

# print("lgr model: ", lgr.coef_, " ", lgr.intercept_)
# print("simp model: ", simp)
# print("alpha: ", alpha)
