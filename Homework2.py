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
import copy

with open('binary_datasets/nltcs/nltcs.train.data', "r") as file:
    reader = csv.reader(file, delimiter=',')
    dataset = np.array(list(reader)).astype(float)

#print(dataset.shape) # (shape 16181, 16), so 16181 samples with 16 features each

### Helper functions:

def marginal_distribution(X, u):
    """
    Return marginal dist for u'th features of data points X as a dictionary of frequencies
    """
    values = defaultdict(float) # standard frequency of 0
    s = 1 / len(X)
    for x in X: # count occurrences (divided by #samples) of this value for feature u
        values[x[u]] += s
    return values

def marginal_pair_distribution(X, u, v):
    """
    Return the marginal distribution for the u'th and v'th features of the data points, X.
    """
    # order features by the order they occur in in the original data
    if u > v:
        u, v = v, u
    values = defaultdict(float) # standard frequency of 0
    s = 1. / len(X)
    for x in X: # count occurrences (divided by #samples) of these values for features u and v
        values[(x[u], x[v])] += s
    return values

def calculate_MI(X, u, v):
    """
    :Param X: data points
    :Param u & v: indices of features to calculate MI for
    """
    # double check that we're filling in the right half of the matrix
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
        self.n = len(data[0]) # amount of features
        self.D = len(data) # amount of samples
        # Set mutual information to a zeros "matrix" to fill in later
        self.MI = np.zeros(shape = (len(data.T), len(data.T)))
        # Compute mutual information
        for v in range(self.n):
            for u in range(v): # only fill in half the matrix since it is symmetrical
                MI_uv = calculate_MI(dataset, u, v)

                self.MI[u][v] = MI_uv

        self.MST = minimum_spanning_tree(-self.MI)
        # If root is None, select random node in data to be root
        if root == None:
            root = np.random.randint(self.MST.shape[0]-1, size=1)
        self.T = breadth_first_order(self.MST, root, False, True)
        self.tree = self.T[1] # list of predecessors of each node, parent of node i is given by self.tree[i]
        self.order = self.T[0] # breadth-first list of nodes
        self.tree[self.tree==-9999] = -1 # set parent to -1 if the node has no parent
        self.lp = self.get_log_params()
    
    def single_prob(self,Z,z,dataset):
        """
        :Param Z: index of parameter
        :Param z: value of parameter
        :Param dataset: the dataset for which we calculate the joint probability
        """
        # calculates p(Z=z) with Laplace correction
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
        # calculates p(Y=y,Z=z) with Laplace correction
        s = 0
        for row in dataset:
            if row[Y] == y and row[Z] == z:
                s += 1
        return (self.alpha + s)/(4*self.alpha + self.D)

    def conditional_prob(self,Y,y,Z,z,dataset):
        """
        :Param Y: index of the first parameter
        :Param y: value of the first parameter
        :Param Z: index of the second parameter
        :Param z: value of the second parameter
        :Param dataset: the dataset for which we calculate the joint probability
        """
        # calculates p(Y=y|Z=z) with Laplace correction
        return self.joint_prob(Y, y, Z, z, dataset) / self.single_prob(Z,z, dataset)

    def get_tree(self):
        return self.tree, self.order
    
    def get_log_params(self):
        log_params = np.zeros((len(self.tree), 2,2))
        tr = self.get_tree()[0]
    
        for i in range(len(tr)):
            crn_parent = tr[i]
            if crn_parent == -1:
                log_params[i] = [[np.log(self.single_prob(i, 0, self.data)), np.log(self.single_prob(i, 1, self.data))],
                                [np.log(self.single_prob(i, 0, self.data)), np.log(self.single_prob(i, 1, self.data))]]
            else:
                Y = i
                Z = crn_parent
                log_params[i] = [[np.log(self.conditional_prob(Y, 0, Z, 0, dataset)), np.log(self.conditional_prob(Y, 0, Z, 1, dataset))],
                                [np.log(self.conditional_prob(Y, 1, Z, 0, dataset)), np.log(self.conditional_prob(Y, 1, Z, 1, dataset))]]
        return log_params

    def compute_log_prob(self, obs_rv, obs_ind, unobs_ind):
        """
        :Param obs_rv: values of the observed random variables y
        :Param obs_ind: observed random variables y's indices
        :Param unobs_ind: unobserved random variables z's indices 
        :Return: dictionary which contains as values the joint probabilities p(y,z) for each value z can take (0 or 1) and the fixed, already observed values of y
                 keys are given by a tuple of the values of y and z
        """
        if len(unobs_ind) == 0:
            # No more unobserved indices, so can compute a probability directly from the tree
            # Recall that the parent of node (rv) i is given by self.tree[i]
            # Since we work with log probabilities, the product of the terms in the joint becomes the sum of the logs      
            total_log_prob_joint = 0
            for index in obs_ind:
                value_rv_i = obs_rv[index]
                index_parent_rv_i = self.tree[index]
                value_parent_rv_i = obs_rv[index_parent_rv_i]
                total_log_prob_joint += self.lp[index, value_parent_rv_i, value_rv_i]
            
            # Get the key for the resulting dictionary by ordering the observed rv's by their original order
            vals = list(zip(obs_rv.copy(), obs_ind.copy()))
            vals.sort(key = lambda vals: vals[1])
            obs_rv_ordered = [vals[i][0] for i in range(0, len(vals))]
            return {tuple(obs_rv_ordered): total_log_prob_joint}
        else:
            # Choose one unobserved variable and compute and merge the dictionaries for both its values
        
            # Give the last unobserved variable value 0
            # Be careful to not overwrite the previous lists
            obs_rv_new = obs_rv.copy()
            obs_ind_new = obs_ind.copy()
            unobs_ind_new = unobs_ind.copy()

            obs_rv_new.append(0)
            obs_ind_new.append(unobs_ind[-1])
            unobs_ind_new.pop()
            dict1 = self.compute_log_prob(obs_rv_new, obs_ind_new, unobs_ind_new)

            # Give the unobserved variable value 1
            # Be careful to not overwrite the previous lists
            obs_rv_new = obs_rv.copy()
            obs_ind_new = obs_ind.copy()
            unobs_ind_new = unobs_ind.copy()

            obs_rv_new.append(1)
            obs_ind_new.append(unobs_ind[-1])
            unobs_ind_new.pop()
            dict2 = self.compute_log_prob(obs_rv_new, obs_ind_new, unobs_ind_new)

            # Merge
            dict1.update(dict2)
            return dict1

    def log_prob(self, x, exhaustive:bool=False):
        res = []
        sums_dict = {}
        probs_dict = {}
        if exhaustive:
            for query in x:
                # First get indices of observed and unobserved RVs, and values of the observed indices
                y_rvs_ind = [i for i in range(0, len(query)) if not np.isnan(query[i])] # observed
                y_rvs_vals = [query[i] for i in y_rvs_ind] 
                z_rvs_ind = [i for i in range(0, len(query)) if np.isnan(query[i])] # unobserved

                # Gather all p(y,z) in a dictionary so that we can sum z out z by summing over this list
                # This dictionary contains p(y,z) where y always takes the fixed, observed values but z's values can be 0 or 1
                dict_log_probs = self.compute_log_prob(y_rvs_vals, y_rvs_ind, z_rvs_ind)
                # Explicitly sum out z
                # Note, dict_log_probs contains log(p(y,z)) so take the sum over z of p(y,z) we need to remove the log first and then reapply it after summing
                return np.log(sum([np.exp(log_prob) for log_prob in dict_log_probs.values()]))
        else:
            # Compute Conditional probabilities using message passing
            pass
        return np.array(res)

    def sample(self, nsamples:int):
        pass


CLT = BinaryCLT(dataset)
tree = CLT.get_tree()
T, bfo = build_chow_liu_tree(dataset, len(dataset[0]))
CLT.get_log_params()
print(CLT.log_prob([(0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,np.nan)], exhaustive=True))



#nx.draw(T)
#plt.show()
# pos = graphviz_layout(tree, prog="dot")
# nx.draw_networkx(tree, pos)
# plt.show()