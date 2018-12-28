import numpy as np
import scipy.io
import config

# ----------------------------------------------------------------------------------------------------
#
# RANDOM TREE GENERATION
#
# ----------------------------------------------------------------------------------------------------

import collections
from igraph import *

def random_binary_tree(nodes):
    v_label = map(str, range(nodes))
    G = Graph.Tree(nodes, 2) # 2 stands for children number
    lay = G.layout('rt')

    position = {k: lay[k] for k in range(nodes)}
    Y = [lay[k][1] for k in range(nodes)]
    M = max(Y)

    es = EdgeSeq(G) # sequence of edges
    E = [e.tuple for e in G.es] # list of edges

    return E

    # L = len(position)
    # Xn = [position[k][0] for k in range(L)]
    # Yn = [2*M-position[k][1] for k in range(L)]
    # Xe = []
    # Ye = []
    # for edge in E:
    #     Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    #     Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]
    #
    # labels = v_label

print (random_binary_tree(40))

class UndirectedGraph():
    def __init__(self, vertices):
        self.vertices = vertices
        self.adjacency_list = [[] for _ in range(vertices)]
        self.edges = []

    def add_edge(self, source, dest):
        assert source < self.vertices
        assert dest < self.vertices
        self.adjacency_list[source].append(dest)
        self.adjacency_list[dest].append(source)
        self.edges.append((str(source), str(dest)))

    def get_edge(self, vertex):
        for e in self.adjacency_list[vertex]:
            yield e

    def get_vertex(self):
        for v in range(self.vertices):
            yield v

def unweighted_tree(graph):
    # edges = []
    for i in range(1, graph.vertices):
        graph.add_edge(np.random.randint(i), i)
        # edges.append((np.random.randint(i), i))
    return graph

# g = UndirectedGraph(1000)
# g = unweighted_tree(g)
#
# print(g.edges)

# def weighted_tree(number_nodes):
#     edges = [[0, 1]]
#     for i in range(1, number_nodes+1):
#         edge_choices = [j for j in range(0, i+1)]
#         edge_choices = dict((choice,0) for choice in edge_choices)
#         for edge in edges:
#             edge_choices[edge[0]] += 1
#         probabilities = []
#         print(edge_choices)
#         for freq in list(edge_choices.values()):
#             probabilities.append(abs(.5 - (float(freq) / len(edges))))
#         print(probabilities)
#         edges.append([np.random.choice(list(edge_choices.keys()), p = probabilities), i+1])
#         print (edges)
#     return edges

# ----------------------------------------------------------------------------------------------------
#
# COMPUTE DISCRETE METRIC
#
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
#
# POINCARE EMBEDDING (AND CONVERSION TO HYPERBOLOID)
#
# ----------------------------------------------------------------------------------------------------

from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath

DIMENSION = 2
# file_path = datapath('/Users/muon/Documents/ML/poincare-embeddings/wordnet/mammal_closure.tsv')

# model = PoincareModel(PoincareRelations(file_path), negative=2, size=DIMENSION)
# model.train(epochs=50)
#
# print(model.kv.vectors)


'''
g = UndirectedGraph(1000)
g = unweighted_tree(g)
model = PoincareModel(g.edges, negative=2, size=DIMENSION)
model.train(epochs=200)

print(model.kv.vectors)

from gensim.viz.poincare import poincare_2d_visualization
plot = poincare_2d_visualization(model, g.edges, "Test Graph")

import plotly.plotly as py
import plotly

plotly.tools.set_credentials_file(username=config.username, api_key=config.api_key)
py.plot(plot, filename="test9")

'''


# Parameters
# n x D matrix B (n points, D dimension)
def ball_to_hyperboloid(B):
    x0 = (2. / (1 - np.sum(B, axis=1))) - 1
    x = np.multiply(B, x0 + 1)
    return np.concatenate(x0, x, axis=1)

# ----------------------------------------------------------------------------------------------------
#
# LEARNING M
#
# ----------------------------------------------------------------------------------------------------

minkowski_diagonal = [1 for i in range(DIMENSION)]
minkowski_diagonal[0] = -1
minkowski_metric_tensor = np.diag(minkowski_diagonal)
print(minkowski_metric_tensor)

def hyperboloid_metric(x, y, Q, G):
    modified_metric = np.matmul(np.matmul(Q.T, G), Q)
    return np.arccosh(1 - (.5 * np.matmul(np.matmul((x.T - y.T), modified_metric), (x - y))))

# To make similarity sets (using LABELs):
# Don't double count pairs
# model.kv.__getitem__([similar_node1, similar_node2, ...])

# Require that (x-y)^T Q^T G Q (x - y) > 0 otherwise we don't have a true sense of distance (ie want PSD Q)

def loss_function(Q):
    total = 0
    for i in S:
        total += hyperboloid_metric(i[0], i[1], Q, minkowski_metric_tensor)

    for j in D:
        total -= hyperboloid_metric(j[0], j[1], Q, minkowski_metric_tensor)

    return total

def learn_Q(Q_initial, learning_rate, precision, max_iters, grad):
    iters = 0
    prev_step = 1
    cur_Q = Q_initial
    while prev_step > precision and iters < max_iters:
        prev_Q = cur_Q #Store current x value in prev_x
        cur_Q = cur_Q - learning_rate * grad(prev_Q) #Grad descent
        prev_step = abs(cur_Q - prev_Q) #Change in x
        iters = iters+1 #iteration count

    return cur_Q

# print( learn_Q(Q_initial, 0.01, 0.000001, 100000, grad(loss_function)) )

# ----------------------------------------------------------------------------------------------------
#
# RECONSTRUCT TREE
#
# ----------------------------------------------------------------------------------------------------




# polblogs1 = scipy.io.loadmat('./data/realnet/polblogs_data_1.mat')
# print(polblogs1.keys())
