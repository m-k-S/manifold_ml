import numpy as np
import scipy.io
from config import username, api_key
import plotly

plotly.tools.set_credentials_file(username=username, api_key=api_key)

# ----------------------------------------------------------------------------------------------------
#
# POINCARE EMBEDDING (AND CONVERSION TO HYPERBOLOID)
#
# ----------------------------------------------------------------------------------------------------

from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath
from gensim.viz.poincare import poincare_2d_visualization

DIMENSION = 2

def plot_embedding(model, labels, title, filename):
    plot = poincare_2d_visualization(model, labels, title)
    py.plot(plot, filename=filename)

# file_path = datapath('')

# model = PoincareModel(PoincareRelations(file_path), negative=2, size=DIMENSION)
# model.train(epochs=50)
#
# print(model.kv.vectors)

tree, edges, labels = random_tree(25, 2)
model = PoincareModel(edges, negative=2, size=DIMENSION)
model.train(epochs=200)

# print(model.kv.vectors)
print(edges)
print(list(labels))

# plot_tree(tree, 25)
# plot_embedding(model, list(labels), "Test Graph", "test10")


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



# polblogs1 = scipy.io.loadmat('./data/realnet/polblogs_data_1.mat')
# print(polblogs1.keys())
