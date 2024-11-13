import pandas as pd
import numpy as np
import math

logv = np.vectorize(math.log)

class emclassifier:

    def __init__(self,attr,cl,model,labels):
        self.attr = attr
        self.cl = cl
        self.model = model
        self.alpha = 0.95
        self.labels = list(labels)
        self.anomal = np.array(len(labels)*[1/len(labels)])
        self.weights =  np.ones(cl.shape[0])
        self.ind = [self.labels.index(x) for x in self.cl]


    def computeprlr(self):
        matrix = self.model.predict_proba(self.attr)
        res =  np.choose(self.ind, matrix.T)
        
        return res
        
    def rcanomal(self,s=2):
        counts = np.bincount(self.ind,weights=1-self.weights) + s/len(self.anomal)

        x = counts.sum()
        counts = counts/x

        print(counts)
        
        return counts

    def fit(self,eps = 0.5):
    

        probslr = self.computeprlr()
        probssi =  np.choose(self.ind, self.anomal)

        den = (self.alpha*probslr + (1-self.alpha)*probssi)
        
        self.weights = self.alpha*probslr/den
        oldlike = logv(den).sum()


        while True:
            self.alpha = np.average(self.weights)
            print(self.alpha)
            print(oldlike)

            self.model.fit(self.attr,self.cl, sample_weight = self.weights)
            self.anomal = self.rcanomal()
            probslr = self.computeprlr()
            probssi =  np.choose(self.ind, self.anomal)
           
            den = (self.alpha*probslr + (1-self.alpha)*probssi)
            newlike = logv(den).sum()
            if abs(newlike-oldlike)<  eps:
                break
            else:
                oldlike = newlike
        
        
            self.weights = self.alpha*probslr/den


    def predict_proba(self,data):
        proba = self.model.predict_proba(data)
        probb = self.anomal
        res = proba*self.alpha + probb*(1-self.alpha)

        return res

    def probanormal(self,data,cl):
        matrix = self.model.predict_proba(data)
        ind = [self.labels.index(x) for x in cl]
        reslr =  np.choose(ind, matrix.T)
        ressi =  np.choose(ind, self.anomal)
        den =(self.alpha*reslr + (1-self.alpha)* ressi)
        return self.alpha*reslr/den, den
        


            
        



        
