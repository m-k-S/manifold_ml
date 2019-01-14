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

# matlab = matlab.engine.start_matlab()
#
# # Node types: pref, unif, bal
# # Label types: unif, hier
# def generate_tree(nodes, type, label):
#     return matlab.gen_rand_tree(nodes, type, 2, label, nargout=4)
#
# Tree = generate_tree(20, 'pref', 'hier')
# Edges = [[str(int(node)) for node in edge] for edge in Tree[0]]
# Labels = [int(label[0]) for label in Tree[3]]

# print(Labels)
# print(Edges)

# ZACHARY'S KARATE CLUB DATASET

intEdges = [[2, 1], [3, 1], [3, 2],[4, 1], [4, 2], [4, 3],
[5, 1],
[6, 1],
[7, 1], [7, 5], [7, 6],
[8, 1], [8, 2], [8, 3], [8, 4],
[9, 1], [9, 3],
[10, 3],
[11, 1], [11, 5], [11, 6],
[12, 1],
[13, 1], [13, 4],
[14, 1], [14, 2], [14, 3], [14, 4],
[17, 6], [17, 7],
[18, 1], [18, 2],
[20, 1], [20, 2],
[22, 1], [22, 2],
[26, 24], [26, 25],
[28, 3], [28, 24], [28, 25],
[29, 3],
[30, 24], [30, 27],
[31, 2], [31, 9],
[32, 1], [32, 25], [32, 26], [32, 29],
[33, 3], [33, 9], [33, 15], [33, 16], [33, 19], [33, 21], [33, 23], [33, 24], [33, 30], [33, 31], [33, 32],
[34, 9], [34, 10], [34, 14], [34, 15], [34,16], [34, 19], [34, 20], [34, 21], [34, 23], [34, 24], [34, 27], [34, 28], [34, 29], [34, 30], [34, 31], [34, 32], [34, 33]]

Labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Edges = [[str(i) for i in Edge] for Edge in intEdges]

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
# REAL DATA

# polblogs1 = scipy.io.loadmat('./data/realnet/polblogs_data_1.mat')
# print(polblogs1.keys())
'''

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
# DEAR LORD! TOO MUCH MAGIC!

# model = PoincareModel(Edges, negative=10, size=DIMENSION)
# model.train(epochs=600)
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

# B = model.kv.vectors
# B = b2h_Matrix(B)
# print(B)

# B = [np.asarray(i[1:]) for i in B]

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

S = []
# for edge in S_Labels:
    # x = b2h_Vector(model.kv.__getitem__(edge[0]))
    # y = b2h_Vector(model.kv.__getitem__(edge[1]))
    # S.append([x, y])
    # S.append([B[edge[0]], B[edge[1]]])

D = []
# for edge in D_Labels:
    # x = b2h_Vector(model.kv.__getitem__(edge[0]))
    # y = b2h_Vector(model.kv.__getitem__(edge[1]))
    # D.append([x, y])
    # D.append([B[edge[0]], B[edge[1]]])


lip = lambda x, y : -x[0] * y[0] + sum([x[i] * y[i] for i in range(1, len(x))])
dhyp = lambda x, y : np.arccosh(-lip(x, y))

# print ("Distance between pair 1 on Poincare ball: " + str(model.kv.distance(S_Labels[0][0], S_Labels[0][1])))
# print ("Distance between pair 1 on hyperboloid: " + str(dhyp(S[0][0], S[0][1])))

# ----------------------------------------------------------------------------------------------------
#
# EUCLIDEAN MDS FOR COMPARISON
#
# ----------------------------------------------------------------------------------------------------

from sklearn.manifold import MDS
import networkx as nx

G = nx.Graph()
G.add_edges_from(intEdges)
discrete_metric = [[0 for _ in range(34)] for _ in range(34)]
for i in range(34):
    for j in range(34):
        discrete_metric[i][j] = len(nx.shortest_path(G, i+1, j+1))
print(discrete_metric)
MDS_embedding = MDS(n_components=2, dissimilarity='precomputed')
graph_embedded = MDS_embedding.fit_transform(discrete_metric)

# ----------------------------------------------------------------------------------------------------
#
# LEARNING M
#
# ----------------------------------------------------------------------------------------------------

from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import quad

minkowski_diagonal = [1 for _ in range(DIMENSION+1)]
minkowski_diagonal[0] = -1
minkowski_metric_tensor = np.diag(minkowski_diagonal)

##########  FUNCTIONS FOR HYPERBOLIC MANIFOLD ####################################
def hyp_mfd(x):
    x0 = np.sqrt(1 + np.sum(np.power(x, 2)))
    x = np.concatenate(([x0], x))

    return x   # hyperbolic map

def hyp_mfd_dist(x,y):
    x0 = x[0]
    y0 = y[0]
    xtail = x[1:]
    ytail = y[1:]

    xGy = np.matmul(xtail.T,ytail) - x0*y0
    return np.arccosh(-xGy)


##########  FUNCTIONS FOR EUCLIDEAN MANIFOLD ####################################
def euclid_mfd(x):
    return x   # its the identity map
def euclid_mfd_dist(x,y):
    return np.linalg.norm(x-y)
###################################################################
def map_dataset_to_mfd(B, Q, mfd_generic):
    FQB = []
    for x in B:
        # x = x[1:]  <<<< DO NOT UNCOMMENT!
        Qx = np.matmul(Q, x)
        mQx = mfd_generic(Qx)
        FQB.append(mQx)

    return FQB

def get_all_neighbors_of(FQx, label_of_FQx, FQB, labels, radius, k, mfd_dist_generic):
    true_neighbors = []
    imposter_neighbors = []

    for idx, x in enumerate(FQB):
        label_x = labels[idx]
        if mfd_dist_generic(FQx, x) < radius:
            if label_x == label_of_FQx:
                true_neighbors.append(x)
            else:
                imposter_neighbors.append(x)

    return true_neighbors, imposter_neighbors

#############################################################################
##################  LMNN for general manifolds
#############################################################################
def lmnn_loss_generic(Q, radius, k, reg, mfd_generic, mfd_dist_generic, B, labels):
    Q = Q.reshape(DIMENSION, DIMENSION)
    print(Q)
    total = 0
    FQB = map_dataset_to_mfd(B, Q, mfd_generic)
    for idx, FQx in enumerate(FQB):
        label_of_FQx = labels[idx]

        FQy_nbrs, FQz_nbrs = get_all_neighbors_of(FQx, label_of_FQx, FQB, labels, radius, k, mfd_dist_generic)

        for FQy in FQy_nbrs:
            total += (1 - reg) * mfd_dist_generic(FQx, FQy)

            for FQz in FQz_nbrs:
                if mfd_dist_generic(FQx,FQz) < mfd_dist_generic(FQx,FQy)+1:  # +1 is the margin
                    total += reg * (1 + mfd_dist_generic(FQx, FQy) - mfd_dist_generic(FQx, FQz))

    return total
################################################
################################################
################################################
def get_sim_dis_pairs(labels):
    sim_idxs = []
    dis_idxs = []
    for i in range(len(labels)): # this is from 0 to len(labels) - 1 YES
        for j in range(i+1, len(labels)): # CHECK SYNTAX   this i want   i+1 to len(labels)-1 looks fine
            if labels[i] == labels[j]:
                sim_idxs.append( [i, j] )
            else:
                dis_idxs.append( [i, j] )

    return sim_idxs, dis_idxs

################################################
def mmc_loss_generic(Q, reg, mfd_generic, mfd_dist_generic, B, labels):
    Q = Q.reshape(DIMENSION, DIMENSION)
    print(Q)
    total = 0
    FQB = map_dataset_to_mfd(B, Q, mfd_generic)
    sim_idxs, dis_idxs = get_sim_dis_pairs(labels)  # sim_idxs should be n x 2,    dis_idxs should be m x 2

    # LIST THE sim_idxs, ndis_idxs MAGIC!! and let me know the result
    #print(sim_idxs)
    #print(dis_idxs)

    nsim_idxs = len(sim_idxs)   # want the row size
    ndis_idxs = len(dis_idxs)  # want the row size

    #SCALE = 1e-50
    for sim_idx in sim_idxs:  # CHECK SYNTAX, ITERATE OVER ROWS OF nsim_idxs
        total += (1-reg) * mfd_dist_generic(FQB[sim_idx[0]], FQB[sim_idx[1]]) / nsim_idxs  #

    for dis_idx in dis_idxs:  # CHECK SYNTAX, ITERATE OVER ROWS OF nsim_idxs
        total -= (reg)   * mfd_dist_generic(FQB[dis_idx[0]], FQB[dis_idx[1]]) / ndis_idxs    # YES, MINUS IS CORRECT

    return total

###########################################################################################
###########################################################################################

import random

def assign_k_random_points_from_fqb(FQB, k):
    selection = []
    for _ in range(k):
        selection.append(random.choice(FQB))
    return selection

###########################################################################################

def kmeans_randomly_partition_data(FQB, k):
    assigned_labels = []
    for _ in FQB:
        assigned_labels.append(random.choice(range(k)))
    return assigned_labels



###########################################################################################


#def kmeans_swap_cost(FQB, assigned_labels, current_cost, idxx, proposed_label_x, mfd_dist_generic):
#    new_cost = current_cost
#
#    curr_label_x = assigned_labels[idxx]
#
#    curr_sum
#    return new_cost

def kmeans_cost_of_assignment(FQB, assigned_labels, mfd_dist_generic, k):
    total_cost = 0
    pts_per_clust = [0 for _ in range(k)]
    for lblx in assigned_labels:  ## this is what i want
        pts_per_clust[lblx] += 1

    for idxx, FQx in enumerate(FQB):
        for idxy, FQy in enumerate(FQB):
            if assigned_labels[idxx] == assigned_labels[idxy]:
                total_cost += mfd_dist_generic(FQx,FQy) / (2*pts_per_clust[assigned_labels[idxx]])

    return total_cost

###########################################################################################

def kmeans_generic(FQB, k, mfd_dist_generic):
    assigned_labels = kmeans_randomly_partition_data(FQB, k)

    converged = False
    while not converged:
        converged = True
        for idxx, FQx in enumerate(FQB):
            curr_cost = kmeans_cost_of_assignment(FQB, assigned_labels, mfd_dist_generic, k)

            min_new_cost = curr_cost
            min_new_label_x = assigned_labels[idxx]

            for kidx in range(k):
                new_labels = assigned_labels
                new_labels[idxx] = kidx
                new_proposed_cost = kmeans_cost_of_assignment(FQB, new_labels, mfd_dist_generic, k)

                if new_proposed_cost < min_new_cost:
                    min_new_cost = new_proposed_cost
                    min_new_label_x = kidx
                    converged = False

            assigned_labels[idxx] = min_new_label_x

    return assigned_labels


###########################################################################################
###########################################################################################
###########################################################################################


Q0 = np.diag([1 for _ in range(DIMENSION)])
# Q0 = minkowski_metric_tensor

# res_NelderMead = minimize(Loss2, Q0, method='nelder-mead', options={'xtol': 1e-3, 'disp': True})
# print(res_NelderMead)

# res_BFGS = minimize(lmnn_Loss, Q0, args=(100, 1, 0.01, deuc), method='BFGS', options={'disp': True})
# print(res_BFGS)
# FOR TESTING PURPOSES CALL THE EUCLI

# res_Powell = minimize(lmnn_loss_generic, Q0, args=(100, 5, 0.5, hyp_mfd, hyp_mfd_dist, B, Labels), method='Powell', options={'disp': True})
# print(res_Powell)

sq_euclid_mfd_dist = lambda x, y : np.linalg.norm(x - y) ** 2
# Centers = kmeans_generic(map_dataset_to_mfd(B, Q0, euclid_mfd), 3, sq_euclid_mfd_dist)



# res_Powell = minimize(mmc_loss_generic, Q0, args=(0.5, euclid_mfd, euclid_mfd_dist, B, Labels), method='Powell', options={'disp': True})
# Qnew = res_Powell.x.reshape(DIMENSION, DIMENSION)
# print(res_Powell)
# print(np.matmul(Qnew.T, Qnew))

# ----------------------------------------------------------------------------------------------------
#
# PLOT DATA
#
# ----------------------------------------------------------------------------------------------------

Initial_Data = B


# ----------------------------------------------------------------------------------------------------
#
# COMPUTE DISCRETE METRIC
#
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
#
# RECONSTRUCT TREE
#
# ----------------------------------------------------------------------------------------------------
