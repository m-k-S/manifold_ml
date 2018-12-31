import numpy as np
import scipy.io
from config import username, api_key
import plotly

plotly.tools.set_credentials_file(username=username, api_key=api_key)

# ----------------------------------------------------------------------------------------------------
#
# RANDOM TREE GENERATION AND VISUALIZATION
#
# ----------------------------------------------------------------------------------------------------

import collections
from igraph import *
import plotly.plotly as py
import plotly.graph_objs as go

def random_tree(nodes, branching_factor):
    labels = [str(i) for i in range(nodes)]
    G = Graph.Tree(nodes, branching_factor)
    lay = G.layout('rt')

    es = EdgeSeq(G) # sequence of edges
    E = [e.tuple for e in G.es] # list of edges

    return G, E, list(labels)

def plot_tree(tree, nodes, labels):
    lay = tree.layout('rt')
    es = EdgeSeq(tree) # sequence of edges
    E = [e.tuple for e in tree.es] # list of edges

    position = {k: lay[k] for k in range(nodes)}
    Y = [lay[k][1] for k in range(nodes)]
    M = max(Y)

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]


    lines = go.Scatter(x=Xe,
                    y=Ye,
                    mode='lines',
                    line=dict(color='rgb(210,210,210)', width=1),
                    hoverinfo='none'
                    )
    dots = go.Scatter(x=Xn,
                    y=Yn,
                    mode='markers',
                    name='',
                    marker=dict(symbol='dot',
                                size=18,
                                color='#6175c1',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                    text=labels,
                    hoverinfo='text',
                    opacity=0.8
                    )

    labels = list(labels)
    annotations = go.Annotations()
    for k in range(L):
        annotations.append(
            go.Annotation(
                text=labels[k], # or replace labels with a different list for the text within the circle
                x=position[k][0], y=2*M-position[k][1],
                xref='x1', yref='y1',
                font=dict(color='rgb(250,250,250)', size=10),
                showarrow=False)
        )

    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

    layout = dict(title= 'Tree with Reingold-Tilford Layout',
                  annotations=annotations,
                  font=dict(size=12),
                  showlegend=False,
                  xaxis=go.XAxis(axis),
                  yaxis=go.YAxis(axis),
                  margin=dict(l=40, r=40, b=85, t=100),
                  hovermode='closest',
                  plot_bgcolor='rgb(248,248,248)'
                  )

    data=go.Data([lines, dots])
    fig=dict(data=data, layout=layout)
    fig['layout'].update(annotations=annotations)
    py.plot(fig, filename='Tree-Reingold-Tilf')


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
from gensim.viz.poincare import poincare_2d_visualization
from os import getcwd

DIMENSION = 2

# iGRAPH EMBEDDING

def plot_embedding(model, labels, title, filename):
    plot = poincare_2d_visualization(model, labels, title)
    py.plot(plot, filename=filename)

# file_path = datapath('')

# model = PoincareModel(PoincareRelations(file_path), negative=2, size=DIMENSION)
# model.train(epochs=50)

'''
tree, edges, labels = random_tree(100, 2)
model = PoincareModel(edges, negative=10, size=DIMENSION)
model.train(epochs=600)

# print(model.kv.vectors)
print(list(labels))
# print(edges)
edge_labels = []
for edge in edges:
    edge_labels.append([str(edge[0]), str(edge[1])])
# edge_labels = [list(edge) for edge in edges]

plot_tree(tree, 100, labels)
plot_embedding(model, edge_labels, "Test Graph", "test12")
'''

'''
g = UndirectedGraph(200)
g = unweighted_tree(g)
model = PoincareModel(g.edges, negative=10, size=DIMENSION)
model.train(epochs=400)

plot = poincare_2d_visualization(model, g.edges, "Test Graph")

py.plot(plot, filename="Test 14")
'''

# Parameters:
#   n x D matrix B (n points, D dimension) [ie transposed]
# Returns:
#   n x D+1 matrix
def ball_to_hyperboloid(B):
    B = np.asarray(B)
    # x0 is a 1xn dim vector
    x0 = (2. / (1 - np.sum(np.power(B, 2), axis=1))) - 1
    x = np.multiply(B.T, x0 + 1)

    print(x.shape)
    print(x)
    print(x0.shape)
    return np.vstack((x, x0)).T

# D = model.kv.vectors
# D = ball_to_hyperboloid(D)

edges = []
randnet = open(os.getcwd() + "/data/randnet/psmodel_deg4_gma2.25_10_net.txt", "r")
for line in randnet:
    line = line.split("\n")[0]
    line = line.split(" ")
    edges.append([line[0].rstrip(), line[1].rstrip()])

# labels = [str(i) for i in range(1, 501)]
model = PoincareModel(edges, negative=2, size=DIMENSION)
model.train(epochs=200)
plot_embedding(model, edges, "Test Graph", "test14")


# ----------------------------------------------------------------------------------------------------
#
# LEARNING M
#
# ----------------------------------------------------------------------------------------------------

minkowski_diagonal = [1 for _ in range(DIMENSION+1)]
minkowski_diagonal[0] = -1
minkowski_metric_tensor = np.diag(minkowski_diagonal)
print(minkowski_metric_tensor)

'''
def hyperboloid_metric(x, y, Q, G):
    modified_metric = np.matmul(np.matmul(Q.T, G), Q)
    return np.arccosh((.5 * np.matmul(np.matmul((x.T - y.T), modified_metric), (x - y))) - 1)

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
'''

def modified_distance(x, y, Q, G):
    modified_metric = np.matmul(np.matmul(Q.T, G), Q)
    return np.negative(np.matmul(np.matmul(x.T, modified_metric), y.T))

def L(Q):
    total = 0
    for i in S:
        x = modified_distance(i[0], i[1], Q, minkowski_metric_tensor)
        total += 1. / (x + np.sqrt(x**2 - 1))

    for j in D:
        x = modified_distance(j[0], j[1], Q, minkowski_metric_tensor)
        total -= 1. / (x + np.sqrt(x**2 - 1))

    return total

def gradL(Q):
    G = minkowski_metric_tensor
    ip = lambda x, y : np.matmul(x.T, np.matmul(Q.T, np.matmul(G, np.matmul(Q, y))))
    dz = lambda x, y : np.matmul(np.matmul(G, np.matmul(Q, x)), y.T) + np.matmul(np.matmul(G, np.matmul(Q, y)), x.T)
    df = lambda x, y : (-1 / (-ip(x, y) + np.sqrt(ip(x, y) ** 2 - 1)) ** 2) * (-1 + (ip(x, y) / np.sqrt(ip(x, y) ** 2 - 1))) * (dz(x, y))

    for edge in S:


    total = 0


# ----------------------------------------------------------------------------------------------------
#
# RECONSTRUCT TREE
#
# ----------------------------------------------------------------------------------------------------




# polblogs1 = scipy.io.loadmat('./data/realnet/polblogs_data_1.mat')
# print(polblogs1.keys())
