# ----------------------------------------------------------------------------------------------------
#
# MDS
#
# ----------------------------------------------------------------------------------------------------

from sklearn.manifold import MDS
import networkx as nx
import scipy.io
import numpy as np
from scipy.optimize import minimize
from mfd_functions import *

# karate1 = scipy.io.loadmat('./karate_edges_new.mat')
# intEdges = karate1['edges']

trueEdges = []
ptsperclass = 30
Labels = [0 for _ in range(ptsperclass * 20 + 13)]
for idx, cat in enumerate(['alt', 'comp', 'misc', 'rec', 'sci', 'soc', 'talk']):
    trueEdges.append([0, idx])

    if cat == 'alt':
        # Alt -> Atheism
        for i in range(13, 13+ptsperclass):
            trueEdges.append([1, i])
            Labels[i] = 0

    if cat == 'comp':
        # Comp -> Sys
        trueEdges.append([2, 8])

        # Sys -> Mac
        for i in range(13 + ptsperclass, 13 + (2 * ptsperclass)):
            trueEdges.append([8, i])
            Labels[i] = 1

        # Sys -> IBM
        for j in range(13 + (2 * ptsperclass), 13 + (3 * ptsperclass)):
            trueEdges.append([8, j])
            Labels[j] = 2

        # Comp -> OS
        for k in range(13 + (3 * ptsperclass), 13 + (4 * ptsperclass)):
            trueEdges.append([2, k])
            Labels[k] = 3

        # Comp -> Graphics
        for k in range(13 + (4 * ptsperclass), 13 + (5 * ptsperclass)):
            trueEdges.append([2, k])
            Labels[k] = 4

        # Comp -> Windows
        for k in range(13 + (5 * ptsperclass), 13 + (6 * ptsperclass)):
            trueEdges.append([2, k])
            Labels[k] = 5

    if cat == 'misc':
        # Misc -> For Sale
        for i in range(13 + (6 * ptsperclass), 13 + (7 * ptsperclass)):
            trueEdges.append([3, i])
            Labels[i] = 6

    if cat == 'rec':
        # Rec -> Sport
        trueEdges.append([4, 9])

        # Sport -> Baseball
        for i in range(13 + (7 * ptsperclass), 13 + (8 * ptsperclass)):
            trueEdges.append([9, i])
            Labels[i] = 7

        # Sport -> Hockey
        for j in range(13 + (8 * ptsperclass), 13 + (9 * ptsperclass)):
            trueEdges.append([9, j])
            Labels[j] = 8

        # Rec -> Autos
        for k in range(13 + (9 * ptsperclass), 13 + (10 * ptsperclass)):
            trueEdges.append([4, k])
            Labels[k] = 9

        # Rec -> Motorcycles
        for k in range(13 + (10 * ptsperclass), 13 + (11 * ptsperclass)):
            trueEdges.append([4, k])
            Labels[k] = 10

    if cat == 'sci':
        # Sci -> Crypt
        trueEdges.append([5, 10])

        # Sci -> Med
        trueEdges.append([5, 11])

        # Crypt -> Nodes
        for i in range(13 + (11 * ptsperclass), 13 + (12 * ptsperclass)):
            trueEdges.append([10, i])
            Labels[i] = 11

        # Med -> Nodes
        for j in range(13 + (12 * ptsperclass), 13 + (13 * ptsperclass)):
            trueEdges.append([11, j])
            Labels[j] = 12

        # Sci -> Electronic
        for k in range(13 + (13 * ptsperclass), 13 + (14 * ptsperclass)):
            trueEdges.append([5, k])
            Labels[k] = 13

        # Sci -> Space
        for k in range(13 + (14 * ptsperclass), 13 + (15 * ptsperclass)):
            trueEdges.append([5, k])
            Labels[k] = 14

    if cat == 'soc':
        # Soc -> Religion
        for i in range(13 + (15 * ptsperclass), 13 + (16 * ptsperclass)):
            trueEdges.append([6, i])
            Labels[i] = 15

    if cat == 'talk':
        # Talk -> Politics
        trueEdges.append([7, 12])

        # Talk -> Religion
        for i in range(13 + (16 * ptsperclass), 13 + (17 * ptsperclass)):
            trueEdges.append([7, i])
            Labels[i] = 16

        # Politics -> Mideast
        for j in range(13 + (17 * ptsperclass), 13 + (18 * ptsperclass)):
            trueEdges.append([12, j])
            Labels[j] = 17

        # Politics -> Guns
        for k in range(13 + (18 * ptsperclass), 13 + (19 * ptsperclass)):
            trueEdges.append([12, k])
            Labels[k] = 18

        # Politics -> Misc
        for k in range(13 + (19 * ptsperclass), 13 + (20 * ptsperclass)):
            trueEdges.append([12, k])
            Labels[k] = 19

G = nx.Graph()
G.add_edges_from(trueEdges)
max_size = G.order()
discrete_metric = [[0 for _ in range(max_size)] for _ in range(max_size)]
for i in range(max_size):
    for j in range(max_size):
        try:
            discrete_metric[i][j] = len(nx.shortest_path(G, i, j)) - 1
        except nx.exception.NetworkXNoPath:
            discrete_metric[i][j] = 50


# print(discrete_metric)
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
            loss += (mfd_dist_generic(FB[i], FB[j], integrand) - Dist[i][j]) ** 2  # no square because of outliers
                                                                                    # stupid me it is being called
    return loss

def mds_initialization(npts, dim):
    pts = []
    mean = np.zeros(dim)
    cov = np.diag(mean)
    for i in range(npts):
        pts.append(np.random.multivariate_normal(mean, cov))

    return np.asarray(pts)

B0 = mds_initialization(max_size, 2)

mds_Powell = minimize(mds_loss, B0, args=(max_size, 2, discrete_metric, hyp_mfd, hyp_mfd_dist, None), method='Powell', options={'disp': True})
print(mds_Powell)
Bnew = mds_Powell.x.reshape(max_size, 2)
print(Bnew)
scipy.io.savemat('20newsgroup_hmds30.mat', mdict = {'arr': Bnew})
