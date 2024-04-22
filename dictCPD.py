from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator,BDeuScore


class dictCPD:

    def __init__(self,variable,variable_card,values,evidence,evidence_card,state_names):
        self.variable = variable 
        self.variable_card = variable_card
        self.values = values
        self.evidence = evidence
        self.evidence_card = evidence_card
        self.state_names = state_names

    def fit(self,variable,network,dataset):
        bayest = BayesianEstimator(model=network, data=dataset, state_names=network.states)

 