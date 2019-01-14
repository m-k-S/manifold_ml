import numpy as np
import scipy.io
from config import username, api_key
import plotly

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
    labels = map(str, range(nodes))
    G = Graph.Tree(nodes, branching_factor)
    lay = G.layout('rt')

    es = EdgeSeq(G) # sequence of edges
    E = [e.tuple for e in G.es] # list of edges

    return G, E, labels

def plot_tree(tree, nodes):
    labels = map(str, range(nodes))
    labels = list(labels)
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
    print(labels)
    print(L)
    print(labels[0])
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
