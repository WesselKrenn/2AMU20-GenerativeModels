import os
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import csv
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
#import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx

with open('binary_datasets/nltcs/nltcs.train.data', "r") as file:
    reader = csv.reader(file, delimiter=',')
    dataset = np.array(list(reader)).astype(np.float)

#print(dataset.shape) # (shape 16181, 16)

### Helper functions:

def marginal_distribution(X, u):
    """
    Return marginal dist for u'th features of data points X
    """
    values = defaultdict(float)
    s = 1 / len(X)
    for x in X:
        values[x[u]] += s
    return values

def marginal_pair_distribution(X, u, v):
    """
    Return the marginal distribution for the u'th and v'th features of the data points, X.
    """
    if u > v:
        u, v = v, u
    values = defaultdict(float)
    s = 1. / len(X)
    for x in X:
        values[(x[u], x[v])] += s
    return values

def calculate_MI(X, u, v):
    """
    :Param X: data points
    :Param u & v: indices of features to calculate MI for
    """
    if u > v:
        u, v = v, u
    marginal_u = marginal_distribution(X, u)
    marginal_v = marginal_distribution(X, v)
    marginal_uv = marginal_pair_distribution(X, u, v)

    I=0.
    for x_u, p_x_u in marginal_u.items():
        for x_v, p_x_v in marginal_v.items():
            if (x_u, x_v) in marginal_uv:
                p_x_uv = marginal_uv[(x_u, x_v)]
                I += p_x_uv * (np.log(p_x_uv) - np.log(p_x_u)- np.log(p_x_v))
    return I

def build_chow_liu_tree(X, n):
    """
    Extra, visualize graph 
    """
    G = nx.Graph()
    for v in range(n):
        G.add_node(v)
        for u in range(v):
            G.add_edge(u, v, weight=-calculate_MI(dataset, u, v))
    T = nx.minimum_spanning_tree(G)
    bfo = breadth_first_order(T, None, False, True)
    return T, bfo

###
# End of helper functions
###

class BinaryCLT:
    def __init__(self, data, root:int=None, alpha:float = 0.01):
        self.alpha = alpha
        self.data = data
        self.n = len(data[0])
        # Set mutual information to a zeros "matrix" to fill in later
        self.MI = np.zeros(shape = (len(data.T), len(data.T)))
        # Compute mutual information
        for v in range(self.n):
            for u in range(v):
                #print(u, v, calculate_MI(dataset, u, v))
                MI_uv = calculate_MI(dataset, u, v)

                self.MI[u][v] = MI_uv

        self.MST = minimum_spanning_tree(-self.MI)
        # If root is None, select random node in data to be root
        if root == None:
            root = np.random.randint(self.MST.shape[0]-1, size=1)
        self.T = breadth_first_order(self.MST, root, False, True)
        self.tree = self.T[1]
        self.order = self.T[0]
        self.tree[self.tree==-9999] = -1

    def gettree(self):
        return self.tree, self.order
    
    def getlogparams(self):
        log_params = np.zeros((len(tree), 2,2))
        
        pass

    def logprob(self, x, exhaustive:bool=False):
        pass

    def sample(self, nsamples:int):
        pass



CLT = BinaryCLT(dataset)
tree = CLT.gettree()

print(tree)
#T, bfo = build_chow_liu_tree(dataset, len(dataset[0]))
#print(bfo)
#pos = graphviz_layout(tree, prog="twopi")
#nx.draw_networkx(tree, pos)
#plt.show()