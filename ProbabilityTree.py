from pgmpy.estimators import BDeuScore
import math


def simplifyfeatures(features,data):
  feat = []
  for x in features:
    if data[x].unique().size > 1:
      feat.append(x)

  return feat







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
  

class probabilitytree:

  def __init__(self):
    self.node = None
    self.frequencies = dict()
    self.chidren = None
    self.model = None
    self.leaf = True



  def fit(self,data, features , target, names, s, double=True):
    frequencies = data[target].value_counts().to_dict()
    h = data[target].unique().size
    feat = simplifyfeatures(features, data)
    if not feat or h<=1:
      self.frequencies = frequencies
      self.node = target
      self.leaf = True
      for x in names[target]:
        if not x in self.frequencies:
          self.frequencies[x] = 0
    else:
      (node,score) = select(data,feat,target,names,s)
    #   print("Uno",node,score)
      if score <=0:
        if double and len(feat)>1:
          feat.remove(node)
      
          (node2,score2)= select(data,feat,target,names,s, [node])
        #   print("dos",node2,score2)
          feat.append(node)
        if not double or len(feat)<=1 or score2<=0:
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
      feat.remove(node)
      for x in names[node]:
        ch = probabilitytree()
        datax = data.loc[data[node] == x]
        ch.fit(datax, feat , target, names, s/len(names[node]))
        self.children[x] = ch
    

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
        self.children[x].updatew(datax,v,states,s/len(self.children.keys()))
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


  

