# ----------------------------------------------------------------------------------------------------
#
# MDS
#
# ----------------------------------------------------------------------------------------------------

from sklearn.manifold import MDS
import networkx as nx
import time
import scipy.io
import numpy as np
from scipy.optimize import minimize
from mfd_functions import *

karate1 = scipy.io.loadmat('./karate_edges_new.mat')
intEdges = karate1['edges']

G = nx.Graph()
G.add_edges_from(intEdges.tolist())
max_size = G.order()
discrete_metric = [[0 for _ in range(max_size)] for _ in range(max_size)]
for i in range(max_size):
    for j in range(max_size):
        try:
            discrete_metric[i][j] = len(nx.shortest_path(G, i+1, j+1)) - 1
        except nx.exception.NetworkXNoPath:
            discrete_metric[i][j] = 50


print(discrete_metric)
MDS_embedding = MDS(n_components=2, dissimilarity='precomputed')
graph_embedded = MDS_embedding.fit_transform(discrete_metric)
scipy.io.savemat('karate_euc.mat', mdict = {'arr': graph_embedded})


# hMDSEdges = []
# for Edge in intEdges.tolist():
#     hMDSEdges.append(str(Edge[0]) + " " + str(Edge[1]))
# print(hMDSEdges)

# print(hMDS_mat)


# model = matlab.hmds(discrete_metric, 2, 10, 1e-5)
# model = b2h_Matrix(np.asarray(model))
# model = h_mds(hMDS_mat, 2, 10, 1e-5)

# print(model)

# scipy.io.savemat('karate_hmds.mat', mdict = {'arr': model})

def mds_loss(B, npts, dim, Dist, mfd_generic, mfd_dist_generic, integrand):
    # Dist is a symmetric npts x npts  distance matrix
    # we are optimizing over location of the points in the base space B

    B = B.reshape(npts, dim)  # datapoints in base space B,  B is the variable of optimization
    # print(B)
    I = np.diag([1 for _ in range(dim)])  # dim x dim identity matrix

    FB = map_dataset_to_mfd(B, I, mfd_generic)

    loss = 0
    for i in range(npts):
        for j in range(i-1):  # traverse the upper triangular matrix
            #loss += (mfd_dist_generic(FB[i], FB[j]) - Dist[i][j])**2  # **2 is supposed to be squared CHECK SYNTAX
            loss += np.abs(mfd_dist_generic(FB[i], FB[j], integrand) - Dist[i][j])  # no square because of outliers

    return loss

def mds_initialization(npts, dim):
    pts = []
    mean = np.zeros(dim)
    cov = np.diag(mean)
    for i in range(npts):
        pts.append(np.random.multivariate_normal(mean, cov))

    return np.asarray(pts)
# 
# B0 = mds_initialization(max_size, 2)
# #
# mds_Powell = minimize(mds_loss, B0, args=(max_size, 2, discrete_metric, hyp_mfd, hyp_mfd_dist, None), method='Powell', options={'disp': True})
# print(mds_Powell)
# Bnew = mds_Powell.x.reshape(max_size, 2)
# print(Bnew)
# scipy.io.savemat('karate_gmds_hyp.mat', mdict = {'arr': Bnew})
