import pandas as pd
import numpy as np
import math

logv = np.vectorize(math.log)


# Class to learn a logistic regression classifier with the possibility that some
# cases are outliers that do not follow a logistic regression. It is assumed that outliers
# do not depend on the attributes (so it is a marginal distribution for the class).
# The fact that a case is an outlier is an unobserved variables

class emclassifier:

    def __init__(self,attr,cl,model,labels):
        self.attr = attr # The attribures
        self.cl = cl  # The class
        self.model = model # The model (logistic regression)
        self.alpha = 0.95 # The initial value of the probability of a non-outlier
        self.labels = list(labels) # The labels of the class variable
        self.anomal = np.array(len(labels)*[1/len(labels)]) # The initial conditional probability of the class for an outlier case (uniform)
        self.weights =  np.ones(cl.shape[0]) # The probability of non-outlier for each one of the cases of the learning sample
        self.ind = [self.labels.index(x) for x in self.cl] # the class values are transformed into numerical indexes (each label into an index value)

    # It predicts the probabilities of the observed class value provided by the logistic regression model

    def computeprlr(self):
        matrix = self.model.predict_proba(self.attr)
        res =  np.choose(self.ind, matrix.T)
        
        return res
        

    # It computes the class distribution for the outliers


    def rcanomal(self,s=2):
        counts = np.bincount(self.ind,weights=1-self.weights) + s/len(self.anomal)

        x = counts.sum()
        counts = counts/x
        
        return counts

    # The em algorithm to compute the probabilities of outliers, the logistic regression for non-outliers and the distribution of outliers

    def fit(self,eps = 0.5):
    

        probslr = self.computeprlr()
        probssi =  np.choose(self.ind, self.anomal)

        den = (self.alpha*probslr + (1-self.alpha)*probssi)
        
        self.weights = self.alpha*probslr/den
        oldlike = logv(den).sum()


        while True:
            self.alpha = np.average(self.weights)
           

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


    # It computes for a dataframe with data of attributes and class values, the probability of not being on outlier for each case

    def probanormal(self,data,cl):
        matrix = self.model.predict_proba(data)
        ind = [self.labels.index(x) for x in cl]
        reslr =  np.choose(ind, matrix.T)
        ressi =  np.choose(ind, self.anomal)
        den =(self.alpha*reslr + (1-self.alpha)* ressi)
        return self.alpha*reslr/den, den
        


    # The probabilities predicted by the model for a new dataframe of test cases

    def predict_proba(self,data):
        proba = self.model.predict_proba(data)
        probb = self.anomal
        res = proba*self.alpha + probb*(1-self.alpha)

        return res

    def score(self,data,cl):
        matrix = self.model.predict_proba(data)
        ind = [self.labels.index(x) for x in cl]
        comp = matrix.argmax(axis=1)
        resp = (ind==comp)
        return np.count_nonzero(resp)/resp.size

        





            
        



        
