import os
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
import csv
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
from itertools import product

with open('binary_datasets/nltcs/nltcs.train.data', "r") as file:
    reader = csv.reader(file, delimiter=',')
    dataset = np.array(list(reader)).astype(float)

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
    all_vals = []
    for edge in T.edges:
        all_vals.append(edge[0])
        all_vals.append(edge[1])
    # building the original adjacency matrix
    adjc_mat = []
    for x in range(max(all_vals)+1):
        row = []
        for y in range(max(all_vals)+1):
            row.append(0)
        adjc_mat.append(row)
    for edge in T.edges:
        adjc_mat[edge[0]][edge[1]] = 1
    graph = csr_matrix(adjc_mat)
    bfo = breadth_first_order(graph, 0, False, True)
    return T, bfo

###
# End of helper functions
###

class BinaryCLT:
    def __init__(self, data, root:int=None, alpha:float = 0.01):
        self.alpha = alpha
        self.data = data
        self.n = len(data[0])
        self.D = len(data)
        # Set mutual information to a zeros "matrix" to fill in later
        self.MI = np.zeros(shape = (len(data.T), len(data.T)))
        # Compute mutual information
        for v in range(self.n):
            for u in range(v):
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
    
    def single_prob(self,Z,z,dataset):
        """
        :Param y: index of the first parameter
        :Param z: index of the second parameter
        :Param dataset: the dataset for which we calculate the joint probability
        """
        # calculates p(Z=z)
        s = 0
        for row in dataset:
            if row[Z] == z:
                s += 1
        return (2*self.alpha + s)/(4*self.alpha + self.D)

    def joint_prob(self,Y,y,Z,z,dataset):
        """
        :Param y: index of the first parameter
        :Param z: index of the second parameter
        :Param dataset: the dataset for which we calculate the joint probability
        """
        # calculates p(Y=y,Z=z)
        s = 0
        for row in dataset:
            if row[Y] == y and row[Z] == z:
                s += 1
        return (self.alpha + s)/(4*self.alpha + self.D)

    def conditional_prob(self,Y,y,Z,z,dataset):
        """
        :Param y: index of the first parameter
        :Param z: index of the second parameter
        :Param dataset: the dataset for which we calculate the joint probability
        """
        # calculates p(Y=y|Z=z)
        return self.joint_prob(Y, y, Z, z, dataset) / self.single_prob(Z,z, dataset)

    def gettree(self):
        return self.tree, self.order
    
    def getlogparams(self):
        log_params = np.zeros((len(self.tree), 2,2))
        tr = self.gettree()[0]
    
        for i in range(len(tr)):
            crn_parent = tr[i]
            if crn_parent == -1:
                log_params[i] = [[np.log(self.single_prob(i, 0, self.data)), np.log(self.single_prob(i, 1, self.data))],[np.log(self.single_prob(i, 0, self.data)), np.log(self.single_prob(i, 1, self.data))]]
            else:
                Y = i
                Z = crn_parent
                log_params[i] = [[np.log(self.conditional_prob(Y, 0, Z, 0, dataset)), np.log(self.conditional_prob(Y, 0, Z, 1, dataset))],[np.log(self.conditional_prob(Y, 1, Z, 0, dataset)), np.log(self.conditional_prob(Y, 1, Z, 1, dataset))]]
        return log_params
    
    def calculate_jpmf(self):
        probs = {}

        for point in self.data:
            t_point = tuple(point)
            if t_point not in probs:
                 probs[t_point]  = 0.0
            else:
                probs[t_point] += 1.0/self.D
        return probs

    def logprob(self, x, exhaustive:bool=False):
        res = 0
        jpmf = self.calculate_jpmf()

    def sample(self, nsamples:int):
        pass


CLT = BinaryCLT(dataset)
tree = CLT.gettree()
T, bfo = build_chow_liu_tree(dataset, len(dataset[0]))
CLT.getlogparams()
CLT.calculate_jpmf()
#nx.draw(T)
#plt.show()
# pos = graphviz_layout(tree, prog="dot")
# nx.draw_networkx(tree, pos)
# plt.show()