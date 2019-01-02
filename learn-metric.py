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
import plotly.plotly as py
import plotly.graph_objs as go
import matlab.engine

matlab = matlab.engine.start_matlab()

# Node types: pref, unif, bal
# Label types: unif, hier
def generate_tree(nodes, type, label):
    return matlab.gen_rand_tree(nodes, type, 2, label, nargout=4)

Tree = generate_tree(20, 'pref', 'hier')
Edges = [[str(int(node)) for node in edge] for edge in Tree[0]]
Labels = [int(label[0]) for label in Tree[3]]

print(Labels)
# print(Edges)

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

def plot_embedding(model, labels, title, filename):
    plot = poincare_2d_visualization(model, labels, title)
    py.plot(plot, filename=filename)

# file_path = datapath('')
# model = PoincareModel(PoincareRelations(file_path), negative=2, size=DIMENSION)
# model.train(epochs=50)

'''
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

'''

model = PoincareModel(Edges, negative=10, size=DIMENSION)
model.train(epochs=600)
# plot_embedding(model, Edges, "Test Graph", "Test 20")

# Parameters:
#   n x D matrix B (n points, D dimension) [ie transposed]
# Returns:
#   n x D+1 matrix
def b2h_Matrix(B):
    B = np.asarray(B)
    # x0 is a 1xn dim vector
    x0 = (2. / (1 - np.sum(np.power(B, 2), axis=1))) - 1
    x = np.multiply(B.T, x0 + 1)

    return np.vstack((x0, x)).T

def b2h_Vector(v):
    x0 = (2. / (1 - np.sum(np.power(v, 2)))) - 1
    x = np.multiply(v, x0 + 1)
    return np.hstack((x0, x))

B = model.kv.vectors
B = b2h_Matrix(B)

S_Labels = []
D_Labels = []
for i in range(1, len(Labels) + 1):
    for j in range(1, len(Labels) + 1):
        if i != j:
            if Labels[i-1] == Labels[j-1]:
                if not ([str(i), str(j)] in S_Labels) and not ([str(j), str(i)] in S_Labels):
                    S_Labels.append([str(i), str(j)])
            else:
                if not ([str(i), str(j)] in D_Labels) and not ([str(j), str(i)] in D_Labels):
                    D_Labels.append([str(i), str(j)])

# print(S)

S = []
for edge in S_Labels:
    x = b2h_Vector(model.kv.__getitem__(edge[0]))
    y = b2h_Vector(model.kv.__getitem__(edge[1]))
    S.append([x, y])

D = []
for edge in D_Labels:
    x = b2h_Vector(model.kv.__getitem__(edge[0]))
    y = b2h_Vector(model.kv.__getitem__(edge[1]))
    D.append([x, y])

# ----------------------------------------------------------------------------------------------------
#
# LEARNING M
#
# ----------------------------------------------------------------------------------------------------

from scipy.optimize import minimize

minkowski_diagonal = [1 for _ in range(DIMENSION+1)]
minkowski_diagonal[0] = -1
minkowski_metric_tensor = np.diag(minkowski_diagonal)

# Require that (x-y)^T Q^T G Q (x - y) > 0 otherwise we don't have a true sense of distance (ie want PSD Q)

def Negatives(x, n):
    samples = []
    for edge in D:
        w = x == edge[0]
        v = x == edge[1]
        if w.all() or v.all():
            samples.append(edge)
        if len(samples) == n:
            break

    return samples

NEGATIVES = 5

def L(Q):
    G = minkowski_metric_tensor
    ip = lambda x, y : np.matmul(x.T, np.matmul(Q.T, np.matmul(G, np.matmul(Q, y))))
    Q = Q.reshape(3, 3)

    total = 0
    for edge in S:
        x = edge[0]
        y = edge[1]

        num = 1. / (-ip(x, y) + np.sqrt(ip(x, y)**2 - 1))

        denom = 0
        samples = Negatives(x, NEGATIVES)
        for sample in samples:
            denom += 1. / (-ip(sample[0], sample[1]) + np.sqrt(ip(sample[0], sample[1])**2 - 1))

        total += num / denom

    return total

def gradL(Q):
    G = minkowski_metric_tensor
    Q = Q.reshape(3, 3)

    ip = lambda x, y : np.matmul(x.T, np.matmul(Q.T, np.matmul(G, np.matmul(Q, y))))
    dz = lambda x, y : np.matmul(np.matmul(G, np.matmul(Q, x)), y.T) + np.matmul(np.matmul(G, np.matmul(Q, y)), x.T)
    df = lambda x, y : (-1 / (-ip(x, y) + np.sqrt(ip(x, y) ** 2 - 1)) ** 2) * (-1 + (ip(x, y) / np.sqrt(ip(x, y) ** 2 - 1))) * (dz(x, y))

    total = 0
    for edge in S:
        x = edge[0]
        y = edge[1]
        num = df(x, y)

        denom = 0
        samples = Negatives(x, NEGATIVES)
        for sample in samples:
            denom += df(sample[0], sample[1])

        total += num / denom

    return total

Q0 = np.diag([1 for _ in range(DIMENSION+1)])
res_NelderMead = minimize(L, Q0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print(res_NelderMead)
# res_BFGS = minimize(L, Q0, method='BFGS', jac=gradL, options={'disp': True})


# ----------------------------------------------------------------------------------------------------
#
# RECONSTRUCT TREE
#
# ----------------------------------------------------------------------------------------------------




# polblogs1 = scipy.io.loadmat('./data/realnet/polblogs_data_1.mat')
# print(polblogs1.keys())
