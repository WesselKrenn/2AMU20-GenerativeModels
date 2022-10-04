from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv
import time

### Start helper functions

def estPriorsJoints(data, alpha):
    """
    Estimate priors and joints prob distributions using MLE
    :param data: The training data
    :param alpha: The Laplace parameter used for smoothing
    :return: The prior, joint distributions
    """
    nFeatures = data.shape[1]
    nSamples = data.shape[0]

    # Calculate priors marginals
    ones_matrix = np.dot(data.T, data)
    ones_diag = np.diag(ones_matrix)
    priors = np.zeros((nFeatures, 2)) # create matrix of (nFeatures, 2) to fill in
    priors[:, 1] = (ones_diag + 2 * alpha) / (nSamples + 4 * alpha)
    priors[:, 0] = 1 - priors[:, 1]

    # Calculate joint marginals
    diag_0 = ones_diag * np.ones((nFeatures, nFeatures))
    diag_1 = diag_0.T
    joints = np.zeros((nFeatures, nFeatures, 2, 2))
    joints[:, :, 0, 0] = nSamples - diag_0 - diag_1 + ones_matrix
    joints[:, :, 0, 1] = diag_0 - ones_matrix
    joints[:, :, 1, 0] = diag_1 - ones_matrix
    joints[:, :, 1, 1] = ones_matrix
    joints = (joints + alpha) / (nSamples + 4 * alpha)

    # Return both to compute MI and log params with
    return priors, joints

def estMI(priors, joints):
    """
    Estimate MI matrix given joints and priors through MLE
    :param priors: prior prob distributions
    :param joints: joint prob distributions
    :return: The MI matrix between every feature i and j
    """
    marginals = np.zeros((priors.shape[0], priors.shape[0], 2, 2))
    marginals[:, :, 0, 0] = np.outer(priors[:, 0], priors[:, 0])
    marginals[:, :, 0, 1] = np.outer(priors[:, 0], priors[:, 1])
    marginals[:, :, 1, 0] = np.outer(priors[:, 1], priors[:, 0])
    marginals[:, :, 1, 1] = np.outer(priors[:, 1], priors[:, 1])

    MI = np.sum(joints * (np.log(joints) - np.log(marginals)), axis=(2, 3))
    np.fill_diagonal(MI, 0)

    return MI

def estLogParams(tree, priors, joints):
    """
    Estimate parameters of tree
    :param tree: The tree as a list (predecessors in tree structure)
    :param priors: Priors dist
    :param joints: Joints dist
    :return: CPTs of each node in the tree
    """
    # initialize log params to fill in
    log_CPT = np.zeros((priors.shape[0], 2, 2))
    root = tree.index(-1)
    features = np.arange(priors.shape[0]).tolist()
    features.remove(root)

    # Make copy of tree and pop root
    parents = tree.copy()
    parents.pop(root)

    # Fill log param table using root, priors and joints
    log_priors = np.log(priors)
    log_CPT[root, 0, 0] = log_CPT[root, 1, 0] = log_priors[root, 0]
    log_CPT[root, 0, 1] = log_CPT[root, 1, 1] = log_priors[root, 1]

    log_joints = np.log(joints)
    log_CPT[features, 0, 0] = log_joints[features, parents, 0, 0] - log_priors[parents, 0]
    log_CPT[features, 1, 0] = log_joints[features, parents, 0, 1] - log_priors[parents, 1]
    log_CPT[features, 0, 1] = log_joints[features, parents, 1, 0] - log_priors[parents, 0]
    log_CPT[features, 1, 1] = log_joints[features, parents, 1, 1] - log_priors[parents, 1]

    # Return final log conditional probabilities table
    return log_CPT

### End helper functions


class BinaryCLT:
    def __init__(self, data, root:int =None, alpha:float = 0.01):
        """
        Initialize class and learn the binary Chow Liu Tree based on given dataset with root and alpha
        :param data: The given (training) dataset
        :param root: The root of the tree, pick random if not given
        :param alpha: The Laplace parameter used for smoothing
        """
        self.nFeatures = data.shape[1]
        self.alpha = alpha
        if root == None:
            self.root = np.random.choice(self.nFeatures)
        else:
            self.root = root
        
        # Compute Mutual information matrix and create Minimum Spanning tree
        priors, joints = estPriorsJoints(data, self.alpha)
        MI_matrix = estMI(priors, joints)
        self.MI = MI_matrix
        MST = minimum_spanning_tree(-MI_matrix)

        # From minium spanning tree and mutual information, do breadth first order to obtain feature order and tree structure
        bfo, tree = breadth_first_order(MST, self.root, False, True)
        
        # Set correct index in tree list to -1 (the root)
        tree[self.root] = -1

        # set bfs to list structure for easier handling
        self.bfo = bfo.tolist()

        # set tree structure
        self.tree = tree.tolist()

        # set log parameters for tree
        self.log_params = estLogParams(self.tree, priors, joints) 

    def get_tree(self):
        return self.tree

    def get_log_params(self):
        return self.log_params

    def log_prob(self, x, exhaustive=False):
        """
        Compute log likelihood of leaf dist given input
        :param x: The given input queries (either fully given or with missing data in between (nan values))
        :param exhaustive: When True, we use exhaustive inference, When False, we use message passing to estimate inference
        :return: The log likelihood of each query x
        """
        queries = x.copy()
        params = self.log_params
        #params_e = np.
        tree = self.tree
        x = np.arange(self.nFeatures).tolist()

        # If query does not contain nan values, exhaustive- and message inference are equal
        if not np.isnan(queries).any():
            log_prob = np.sum(params[x, queries[:, tree], queries[:, x]], axis=1)
            #log_prob = logsumexp(params[x, queries[:, tree], queries[:, x]], axis=1)

        # exhaustive = True
        elif exhaustive:
            log_prob = np.zeros(queries.shape[0])
            # For exhaustive search, get each combination of features and call them "states"
            states = np.array([state for state in itertools.product([0, 1], repeat=self.nFeatures)])
            states_log_probs = np.sum(params[x, states[:, tree], states[:, x]], axis=1)
            #states_log_probs = logsumexp(params[x, states[:, tree], states[:, x]], axis=1)
            for i in range(queries.shape[0]):
                # find index where query[i] is non nan
                idx = np.where(~np.isnan(queries[i]))[0]
                margin_idx = np.where((states[:, idx] == queries[i][idx]).all(axis=1))[0]
                log_prob[i] = logsumexp(states_log_probs[margin_idx])

        else: 
            # Exhaustive is false, message passing implementation
            log_prob = np.zeros(queries.shape[0])
            
            # Loop through queries to compute inference for each
            for query in range(queries.shape[0]):
                # Initialize message as zeros
                messages = np.zeros((self.nFeatures, 2))

                # Using a bottom up approach loop through nodes(fast in our case)
                # And exclude root node (as it has no parent)
                for node in self.bfo[::-1][:-1]:

                    if not np.isnan(queries[query, node]):
                        messages[tree[node], 1] += params[node, 1, int(queries[query, node])] + messages[node, int(queries[query, node])]
                        messages[tree[node], 0] += params[node, 0, int(queries[query, node])] + messages[node, int(queries[query, node])]
                    else:
                        messages[tree[node], 1] += logsumexp([params[node, 1, 0] + messages[node, 0],
                                                              params[node, 1, 1] + messages[node, 1]])
                        messages[tree[node], 0] += logsumexp([params[node, 0, 0] + messages[node, 0],
                                                              params[node, 0, 1] + messages[node, 1]])
            
                # Root is observed
                if not np.isnan(queries[query, self.root]):
                    log_prob[query] = params[self.root, 0, int(queries[query, self.root])] + messages[self.root, int(queries[query, self.root])]

                # Root is not observed
                else:
                    log_prob[query] = logsumexp([params[self.root, 0, 0] + messages[self.root, 0],
                                            params[self.root, 0, 1] + messages[self.root, 1]])
        
        return np.expand_dims(log_prob, axis=1)

    def sample(self, nSamples:int):
        """
        Get a given number of samples from the Chow Liu Tree
        :param nSamples: The number of samples to generate
        :return: the generated samples 
        """
        # Create 2d zeros array with shape (nSamples, nFeatures) to edit during sampling process
        samples = np.zeros((nSamples, self.nFeatures)).astype(np.uint8)

        # Compute root probability
        root_prob = np.exp(self.log_params[self.root, 0, 1])
        # Set root value in samples
        samples[:, self.root] = np.random.binomial(1, root_prob, nSamples)

        # Loop through remainder of features to sample them conditionally of their (grand)parent
        for f in self.bfo[1:]:
            # Get parent and compute its probability
            parent = samples[:, self.tree[f]]
            parent_prob = np.exp(self.log_params[f][parent])[:, 1]
            
            # Add value to samples array
            samples[:, f] = np.random.binomial(1, parent_prob, nSamples)
        
        # return generated samples
        return samples


with open('../binary_datasets/nltcs/nltcs.train.data', "r") as file:
    reader = csv.reader(file, delimiter=',')
    dataset = np.array(list(reader)).astype(float)


def load(path):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        data = np.array(list(reader)).astype(np.float32)
    return data

nltcs_train = load('../binary_datasets/nltcs/nltcs.train.data')
nltcs_test = load('../binary_datasets/nltcs/nltcs.test.data')
nltcs_marginals = load('../nltcs_marginals.data')


CLT = BinaryCLT(data=nltcs_train, root=0, alpha=0.01)


# TODO: code for 2e point 1 to 5.

# task 2e.6
for _ in range(10):
    samples = CLT.sample(1000)
    ll_samples = CLT.log_prob(samples)
    print('Average samples LL: ', np.mean(ll_samples))