import numpy as np
import scipy.io
from config import username, api_key
# import plotly

# plotly.tools.set_credentials_file(username=username, api_key=api_key)

# ----------------------------------------------------------------------------------------------------
#
# RANDOM TREE GENERATION AND VISUALIZATION
#
# ----------------------------------------------------------------------------------------------------

import collections
# import plotly.plotly as py
# import plotly.graph_objs as go
# import matlab.engine

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

polblogs1 = scipy.io.loadmat('./data/realnet/karate_data_1_edges.mat')
intEdges = polblogs1['edges']  ##

Edges = [[str(i) for i in Edge] for Edge in intEdges]

max_size = 0
for Edge in intEdges.tolist():
    for i in Edge:
        if i > max_size:
            max_size = i

# print(Edges)
Labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#                                                         ^
#################################################################################################################

# ZACHARY'S KARATE CLUB DATASET

'''
Labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Edges = [[str(i) for i in Edge] for Edge in intEdges]
'''

# ----------------------------------------------------------------------------------------------------
#
# POINCARE EMBEDDING (AND CONVERSION TO HYPERBOLOID)
#
# ----------------------------------------------------------------------------------------------------

# from gensim.models.poincare import PoincareModel, PoincareRelations
# from gensim.test.utils import datapath
# from gensim.viz.poincare import poincare_2d_visualization
from os import getcwd


def plot_embedding(model, labels, title, filename):
    plot = poincare_2d_visualization(model, labels, title)
    py.plot(plot, filename=filename)

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
# model.train(epochs=50)
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

# scipy.io.savemat('polblogs_hyp.mat', mdict = {'arr': B})

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
import time


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

# print(discrete_metric)
# MDS_embedding = MDS(n_components=2, dissimilarity='precomputed')
# graph_embedded = MDS_embedding.fit_transform(discrete_metric)
# scipy.io.savemat('polblogs_euc.mat', mdict = {'arr': graph_embedded})

# print(discrete_metric)

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


# ----------------------------------------------------------------------------------------------------
#
# LEARNING M
#
# ----------------------------------------------------------------------------------------------------

from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import quad



##########  FUNCTIONS FOR HELICOID MANIFOLD ####################################
def helicoid_mfd(x):
    r = x[0]
    t = x[1]
    mp = [r*np.cos(t), r*np.sin(t), t]
    return mp   # helicoid map

def helicoid_mfd_dist(x,y,integrand):
    eps = 1e-5

    xr = x[0] / np.cos(x[2])
    # xtemp = x[1] / np.sin(x[2])
    # assert(np.abs(xr - xtemp) < eps)
    xs = x[2]

    yr = y[0]/np.cos(y[2])
    # ytemp = y[1] / np.sin(y[2])
    # assert(np.abs(yr - ytemp) < eps)
    ys = y[2]

    bx = np.asarray([np.abs(xr), xs])
    by = np.asarray([np.abs(yr), ys])
# #    bx = np.asarray([0, xs])
#     by = np.asarray([0, ys])

    return np.linalg.norm(bx-by)
    #I = np.diag([1 for _ in bx])
    #x = np.asarray(bx)
    #y = np.asarray(by)
    #dist = learn_distance(bx, by, I, integrand, samples=100, segments=7)
    #return dist



##########  FUNCTIONS FOR HYPERBOLIC MANIFOLD ####################################
def hyp_mfd(x):
    x0 = np.sqrt(1 + np.sum(np.power(x, 2)))
    x = np.concatenate(([x0], x))

    return x   # hyperbolic map

def hyp_mfd_dist(x,y, integrand):
    x0 = x[0]
    y0 = y[0]
    xtail = x[1:]
    ytail = y[1:]

    xGy = np.matmul(xtail.T,ytail) - x0*y0
    return np.arccosh(-xGy)


##########  FUNCTIONS FOR EUCLIDEAN MANIFOLD ####################################
def euclid_mfd(x):
    return x   # its the identity map

def euclid_mfd_dist(x,y, integrand):
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

def get_all_neighbors_of(FQx, label_of_FQx, FQB, labels, radius, k, mfd_dist_generic, mfd_integrand):
#############  CORRECT CODE FOR radius
#    true_neighbors = []
#    imposter_neighbors = []
#
#    for idx, x in enumerate(FQB):
#        label_x = labels[idx]
#        if mfd_dist_generic(FQx, x, mfd_integrand) < radius:
#            if label_x == label_of_FQx:
#                true_neighbors.append(x)
#            else:
#                imposter_neighbors.append(x)
#
#    return true_neighbors, imposter_neighbors

################ CODE FOR K
    dst_from_FQx = []
    for idx, FQxi in enumerate(FQB):
        dst_from_FQx.append( mfd_dist_generic(FQx, FQxi, mfd_integrand))

    idx_of_points = np.argsort(np.asarray(dst_from_FQx))  # <<< only need these indices

    #print(FQx)
    #print(FQB)
    #print(dst_from_FQx)
    #print(idx_of_points)
    #print('-------------------')
    true_neighbors = []
    imposter_neighbors = []
    for i in range(1,k+1):
        nidx = idx_of_points[i]      # +1 because ignoring the zero'th index that is supposed to be FQx itself
        label_nxi = labels[nidx]     # this assumes k < length(FQB)
        if label_nxi == label_of_FQx:
            true_neighbors.append(FQB[nidx])
        else:
            imposter_neighbors.append(FQB[nidx])

    #print(true_neighbors)
    #print(imposter_neighbors)
    #print('=========================')
    return true_neighbors, imposter_neighbors

#############################################################################
##################  LMNN for general manifolds
#############################################################################
def sv_constraint(Q):
    u, s, vh = np.linalg.svd(Q)
    s = s.tolist()
    return max(s)

def lmnn_loss_generic(Q, radius, k, reg, mfd_generic, mfd_dist_generic, mfd_integrand, B, labels):
    dim = len(B[0])
    Q = Q.reshape(dim, dim)
    # print(Q)
    total = 0
    FQB = map_dataset_to_mfd(B, Q, mfd_generic)
    for idx, FQx in enumerate(FQB):
        label_of_FQx = labels[idx]

        #FQy_nbrs, FQz_nbrs = get_all_neighbors_of(FQx, label_of_FQx, FQB, labels, radius, k, mfd_dist_generic, mfd_integrand)
        FQy_nbrs, FQz_nbrs = get_all_neighbors_of(B[idx], label_of_FQx, B, labels, radius, k, mfd_dist_generic, mfd_integrand)

        for FQy in FQy_nbrs:
            total += (1 - reg) * mfd_dist_generic(FQx, FQy, mfd_integrand)

            for FQz in FQz_nbrs:
                if mfd_dist_generic(FQx,FQz, mfd_integrand) < mfd_dist_generic(FQx,FQy, mfd_integrand)+1:  # +1 is the margin
                    total += reg * (1 + mfd_dist_generic(FQx, FQy, mfd_integrand) - mfd_dist_generic(FQx, FQz, mfd_integrand))

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
def mmc_loss_generic(Q, reg, mfd_generic, mfd_dist_generic, mfd_integrand, B, labels):
    dim = len(B[0])
    Q = Q.reshape(dim, dim)
    # print(Q)
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
        total += (1-reg) * mfd_dist_generic(FQB[sim_idx[0]], FQB[sim_idx[1]], mfd_integrand) / nsim_idxs  #

    for dis_idx in dis_idxs:  # CHECK SYNTAX, ITERATE OVER ROWS OF nsim_idxs
        total -= (reg)   * mfd_dist_generic(FQB[dis_idx[0]], FQB[dis_idx[1]], mfd_integrand) / ndis_idxs

    return total

###########################################################################################
###########################################################################################

import random

def assign_k_random_points_from_fqb(FQB, k):
    selection = []
    for _ in range(k):
        selection.append(random.choice(FQB))
    return selection

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

def kmeans_cost_of_assignment(FQB, assigned_labels, mfd_dist_generic, mfd_integrand, k):
    total_cost = 0
    pts_per_clust = [0 for _ in range(k)]
    for lblx in assigned_labels:  ## this is what i want
        pts_per_clust[lblx] += 1

    for idxx, FQx in enumerate(FQB):
        for idxy, FQy in enumerate(FQB):
            if assigned_labels[idxx] == assigned_labels[idxy]:
                total_cost += mfd_dist_generic(FQx,FQy, mfd_integrand) / (2*pts_per_clust[assigned_labels[idxx]])

    return total_cost

###########################################################################################

def kmeans_generic(FQB, k, mfd_dist_generic, mfd_integrand):
    assigned_labels = kmeans_randomly_partition_data(FQB, k)

    converged = False
    while not converged:
        converged = True
        for idxx, FQx in enumerate(FQB):
            curr_cost = kmeans_cost_of_assignment(FQB, assigned_labels, mfd_dist_generic, mfd_integrand, k)

            min_new_cost = curr_cost
            min_new_label_x = assigned_labels[idxx]

            for kidx in range(k):
                new_labels = assigned_labels
                new_labels[idxx] = kidx
                new_proposed_cost = kmeans_cost_of_assignment(FQB, new_labels, mfd_dist_generic, mfd_integrand, k)

                if new_proposed_cost < min_new_cost:
                    min_new_cost = new_proposed_cost
                    min_new_label_x = kidx
                    converged = False

            assigned_labels[idxx] = min_new_label_x

    return assigned_labels


# ----------------------------------------------------------------------------------------------------
#
# MANIFOLD MDS
#
# ----------------------------------------------------------------------------------------------------

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

# B0 = mds_initialization(max_size, 2)
# #
# mds_Powell = minimize(mds_loss, B0, args=(max_size, 2, discrete_metric, hyp_mfd, hyp_mfd_dist, None), method='Powell', options={'disp': True})
# print(mds_Powell)
# Bnew = mds_Powell.x.reshape(max_size, 2)
# print(Bnew)
# ----------------------------------------------------------------------------------------------------
#
# PERFORMANCE EVALUATION
#
# ----------------------------------------------------------------------------------------------------

import sklearn.metrics
from learn_dist import integrand_helicoid, learn_distance

def do_cluster_test(train_ratio, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
    npts = len(Bnew_euc)
    dim_euc = len(Bnew_euc[0])
    dim_mfd = len(Bnew_mfd[0])
        # split data into training and testing
    idx_tr = []
    idx_ts = []
    euc_data_tr = []
    euc_data_ts = []
    mfd_data_tr = []
    mfd_data_ts = []
    labels_tr = []
    labels_ts = []
    for i in range(npts):
        if np.random.random() < train_ratio:   ####   CHECK SYNTAX
            idx_tr.append(i)
            euc_data_tr.append(Bnew_euc[i])
            mfd_data_tr.append(Bnew_mfd[i])
            labels_tr.append(true_labels[i])
        else:
            idx_ts.append(i)
            euc_data_ts.append(Bnew_euc[i])
            mfd_data_ts.append(Bnew_mfd[i])
            labels_ts.append(true_labels[i])

            # learn Q using mmc
    Q0_euc = np.diag([1 for _ in range(dim_euc)])
    Q0_mfd = np.diag([1 for _ in range(dim_mfd)])

    # sv_cons = NonlinearConstraint(sv_constraint, 0, 1)
    euc_res_Powell = minimize(mmc_loss_generic, Q0_euc, args=(0.5, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True})
    mfd_res_Powell = minimize(mmc_loss_generic, Q0_mfd, args=(0.5, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='CG', options={'disp': True})
    # mfd_res_Powell = minimize(mmc_loss_generic, Q0_mfd, args=(0.5, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='trust-constr', constraints=[sv_cons], options={'disp': True})

    # euc_res_Powell = minimize(lmnn_loss_generic, Q0_euc, args=(100, 11, 0.5, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True})
    # mfd_res_Powell = minimize(lmnn_loss_generic, Q0_mfd, args=(100, 200, 0.5, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='Powell', options={'disp': True})
    # mfd_res_Powell = minimize(lmnn_loss_generic, Q0_mfd, args=(100, 200, 0.5, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='trust-constr', constraints=[sv_cons], options={'disp': True})
    print(mfd_res_Powell)

    euc_Qnew = euc_res_Powell.x.reshape(dim_euc, dim_euc)
    mfd_Qnew = mfd_res_Powell.x.reshape(dim_mfd, dim_mfd)

    # print (mfd_Qnew)
    # print (np.matmul(mfd_Qnew.T, mfd_Qnew))

    euc_Qdata_ts = map_dataset_to_mfd(euc_data_ts, euc_Qnew, fxn_euc)
    mfd_Qdata_ts = map_dataset_to_mfd(mfd_data_ts, mfd_Qnew, fxn_mfd)


        # run k-means
    K = len(np.unique(true_labels))   # number of unique labels is the value of K in K-means

    euc_lab_ts  = kmeans_generic(euc_data_ts,  K, fxn_euc_dist, None)
    euc_Qlab_ts = kmeans_generic(euc_Qdata_ts, K, fxn_euc_dist, None)
    mfd_lab_ts  = kmeans_generic(mfd_data_ts,  K, fxn_mfd_dist, fxn_integrand)
    mfd_Qlab_ts = kmeans_generic(mfd_Qdata_ts, K, fxn_mfd_dist, fxn_integrand)

        # evaluate k-means results
    err_euc_orig = eval_cluster_quality(labels_ts, euc_lab_ts)
    err_euc_qlrn = eval_cluster_quality(labels_ts, euc_Qlab_ts)
    err_mfd_orig = eval_cluster_quality(labels_ts, mfd_lab_ts)
    err_mfd_qlrn = eval_cluster_quality(labels_ts, mfd_Qlab_ts)

    return err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn


def eval_cluster_quality(true_labels, assigned_labels):
    true_labels = [i[0] for i in true_labels]   # this fixes the porblem? yes, code is running now -- ok

    ARI = sklearn.metrics.adjusted_rand_score(true_labels, assigned_labels)  # adjusted rand index   (higher number is better)
    NMI = sklearn.metrics.normalized_mutual_info_score(true_labels, assigned_labels) # normalized mutual information (higer number is better)
    err = [ARI, NMI]  # using TWO different evaluation metrics   <<<<< CHECK SYNTAX is fine?
    return err

def do_cluster_tests_all(nrounds, train_ratio, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
    err_euc_orig = []
    err_euc_qlrn = []
    err_mfd_orig = []
    err_mfd_qlrn = []

    for r in range(nrounds):
        eeo,eeq,emo,emq = do_cluster_test(train_ratio, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels)
        err_euc_orig.append(eeo)
        err_euc_qlrn.append(eeq)
        err_mfd_orig.append(emo)
        err_mfd_qlrn.append(emq)

    return err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn

############################################################################################################
############################################################################################################
def do_classification_tests_all(nrounds, train_ratio, K, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
    err_euc_orig = []
    err_euc_qlrn = []
    err_mfd_orig = []
    err_mfd_qlrn = []

    for r in range(nrounds):
        eeo,eeq,emo,emq = do_classification_test(train_ratio, K, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels)
        err_euc_orig.append(eeo)
        err_euc_qlrn.append(eeq)
        err_mfd_orig.append(emo)
        err_mfd_qlrn.append(emq)

    return err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn


def do_classification_test(train_ratio, K, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
    npts = len(Bnew_euc)
    dim_euc = len(Bnew_euc[0])
    dim_mfd = len(Bnew_mfd[0])
        # split data into training and testing
    idx_tr = []
    idx_ts = []
    euc_data_tr = []
    euc_data_ts = []
    mfd_data_tr = []
    mfd_data_ts = []
    labels_tr = []
    labels_ts = []
    for i in range(npts):
        if np.random.random() < train_ratio:   ####   CHECK SYNTAX
            idx_tr.append(i)
            euc_data_tr.append(Bnew_euc[i])
            mfd_data_tr.append(Bnew_mfd[i])
            labels_tr.append(true_labels[i])
        else:
            idx_ts.append(i)
            euc_data_ts.append(Bnew_euc[i])
            mfd_data_ts.append(Bnew_mfd[i])
            labels_ts.append(true_labels[i])

            # learn Q using mmc
    Q0_euc = np.diag([1 for _ in range(dim_euc)])
    Q0_mfd = np.diag([1 for _ in range(dim_mfd)])

    # sv_cons = NonlinearConstraint(sv_constraint, 0, 1)
    #euc_res_Powell = minimize(mmc_loss_generic, Q0_euc, args=(0.5, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True})
    #mfd_res_Powell = minimize(mmc_loss_generic, Q0_mfd, args=(0.5, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='CG', options={'disp': True})
    # mfd_res_Powell = minimize(mmc_loss_generic, Q0_mfd, args=(0.5, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='trust-constr', constraints=[sv_cons], options={'disp': True})

    #K = 5
    reg = 0.5
    euc_res_Powell = minimize(lmnn_loss_generic, Q0_euc, args=(None, K, reg, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True})
    # mfd_res_Powell = minimize(lmnn_loss_generic, Q0_mfd, args=(None, K, reg, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='Powell', options={'disp': True})
    # mfd_res_Powell = minimize(lmnn_loss_generic, Q0_mfd, args=(100, 200, 0.5, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='trust-constr', constraints=[sv_cons], options={'disp': True})
    # print(mfd_res_Powell)

    euc_Qnew = euc_res_Powell.x.reshape(dim_euc, dim_euc)
    print(euc_Qnew)
    scipy.io.savemat('karate_Qeuc.mat', mdict = {'arr': map_dataset_to_mfd(Bnew_euc, euc_Qnew, euclid_mfd)})
    # mfd_Qnew = mfd_res_Powell.x.reshape(dim_mfd, dim_mfd)

    # print (mfd_Qnew)
    # print (np.matmul(mfd_Qnew.T, mfd_Qnew))

    euc_Qdata_ts = map_dataset_to_mfd(euc_data_ts, euc_Qnew, fxn_euc)
    mfd_Qdata_ts = map_dataset_to_mfd(mfd_data_ts, mfd_Qnew, fxn_mfd)

    euc_Qdata_tr = map_dataset_to_mfd(euc_data_tr, euc_Qnew, fxn_euc)
    mfd_Qdata_tr = map_dataset_to_mfd(mfd_data_tr, mfd_Qnew, fxn_mfd)


    #    # run k-means
    #K = len(np.unique(true_labels))   # number of unique labels is the value of K in K-means

    euc_lab_ts   = knnclassify_generic(euc_data_ts,  K, euc_data_tr,  labels_tr, fxn_euc_dist, None)
    euc_Qlab_ts  = knnclassify_generic(euc_Qdata_ts, K, euc_Qdata_tr, labels_tr, fxn_euc_dist, None)
    mfd_lab_ts   = knnclassify_generic(mfd_data_ts,  K, mfd_data_tr,  labels_tr, fxn_euc_dist, None)
    mfd_Qlab_ts  = knnclassify_generic(mfd_Qdata_ts, K, mfd_Qdata_tr, labels_tr, fxn_euc_dist, None)

    # euc_Qlab_ts = kmeans_generic(euc_Qdata_ts, K, fxn_euc_dist, None)
    # mfd_lab_ts  = kmeans_generic(mfd_data_ts,  K, fxn_mfd_dist, fxn_integrand)
    # mfd_Qlab_ts = kmeans_generic(mfd_Qdata_ts, K, fxn_mfd_dist, fxn_integrand)

        # evaluate classification results
    err_euc_orig = eval_classification_quality(labels_ts, euc_lab_ts)
    err_euc_qlrn = eval_classification_quality(labels_ts, euc_Qlab_ts)
    err_mfd_orig = eval_classification_quality(labels_ts, mfd_lab_ts)
    err_mfd_qlrn = eval_classification_quality(labels_ts, mfd_Qlab_ts)

    return err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn

def eval_classification_quality(labels_ts, mfd_lab_ts):
    err_01 = 0
    for i in range(len(labels_ts)):
        if labels_ts[i][0] != mfd_lab_ts[i]:
            err_01 += 1

    err_01 /= len(labels_ts)
    return err_01

def knnclassify_generic(data_ts,  K, data_tr, labels_tr, mfd_dist_generic, mfd_integrand):
    labels_ts = []

    for x_ts in data_ts:
        dst_to_xts = []
        for idx_tr, x_tr in enumerate(data_tr):
            dst_to_xts.append( mfd_dist_generic(x_tr, x_ts, mfd_integrand))

        idx_of_points = np.argsort(np.asarray(dst_to_xts))  # <<< only need these indices

        l = [];
        for i in idx_of_points[:K]:
            l.append(labels_tr[i])
        l_xts = scipy.stats.mode(l)
        labels_ts.append(l_xts[0].tolist()[0])

    return labels_ts
# ----------------------------------------------------------------------------------------------------
#
# MAIN
#
# ----------------------------------------------------------------------------------------------------



# datasetname = 'helicoid'
datasetname = 'karate'
# datasetname = 'football'
# datasetname = 'polbooks'
# datasetname = 'polblogs'

if datasetname == 'karate':
    Beuc   = scipy.io.loadmat('./karate_euc.mat')['arr']
    Bhyp   = scipy.io.loadmat('./karate_gmds_hyp.mat')['arr'] # this file has correct data in it!
    Labels = scipy.io.loadmat('./karate_data_1.mat')['label']
    train_ratio = 0.6
    fxn_mfd = hyp_mfd
    fxn_mfd_dist = hyp_mfd_dist
    #fxn_integrand = integrand_hyp
    fxn_integrand = None

elif datasetname == 'helicoid':
    Beuc = scipy.io.loadmat('./helicoid.mat')['data']  # 3 D
    Bhyp   = scipy.io.loadmat('./helicoid.mat')['base'] # 2 D
    Labels = scipy.io.loadmat('./helicoid.mat')['labels']
    train_ratio = 0.6
    fxn_mfd = helicoid_mfd
    fxn_mfd_dist = helicoid_mfd_dist
    fxn_integrand = integrand_helicoid

elif datasetname == 'football':
    Beuc   = scipy.io.loadmat('./football_euc.mat')['arr']
    Bhyp   = scipy.io.loadmat('./football_gmds_hyp.mat')['arr'] # this file has correct data in it!
    Labels = scipy.io.loadmat('./football_data_1.mat')['label']
    train_ratio = 0.6
    fxn_mfd = hyp_mfd
    fxn_mfd_dist = hyp_mfd_dist
    #fxn_integrand = integrand_hyp
    fxn_integrand = None

elif datasetname == 'polbooks':
    Beuc   = scipy.io.loadmat('./polbooks_euc.mat')['arr']
    Bhyp   = scipy.io.loadmat('./polbooks_gmds_hyp.mat')['arr'] # this file has correct data in it!
    Labels = scipy.io.loadmat('./polbooks_data_1.mat')['label']
    train_ratio = 0.7
    fxn_mfd = hyp_mfd
    fxn_mfd_dist = hyp_mfd_dist
    #fxn_integrand = integrand_hyp
    fxn_integrand = None

elif datasetname == 'polblogs':
    Beuc   = scipy.io.loadmat('./polblogs_euc.mat')['arr']
    Bhyp   = scipy.io.loadmat('./polblogs_gmds_hyp.mat')['arr']  # FILE DOES NOT EXIST!  CORRESPONDING .txt doesnt exist (because the run was freezing)
    Labels = scipy.io.loadmat('./polblogs_data_1.mat')['label']
    train_ratio = 0.8
    fxn_mfd = hyp_mfd
    fxn_mfd_dist = hyp_mfd_dist
    #fxn_integrand = integrand_hyp
    fxn_integrand = None

else:
    print('undefined dataset!')
    assert(1==0)


nrounds = 10

fxn_euc = euclid_mfd
fxn_euc_dist = euclid_mfd_dist


# err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn = do_cluster_tests_all(nrounds, train_ratio, Beuc, fxn_euc, fxn_euc_dist, Bhyp, fxn_mfd, fxn_mfd_dist, fxn_integrand, Labels)
err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn = do_classification_tests_all(nrounds, train_ratio, 1, Beuc, fxn_euc, fxn_euc_dist, Bhyp, fxn_mfd, fxn_mfd_dist, fxn_integrand, Labels)

#
# Q = [[  1.,           0.        ],
#     [-72.84578928,   0.9476725 ]]

# Q = [ [1.94754533e+05,  9.79474623e+05], [-1.15952005e+08, -2.41173469e+06]]
# Q = [ [42.5277812 , -55.44849986], [-18.23708945,  23.76893528]]


# scipy.io.savemat('helicoid_Q.mat', mdict = {'arr': map_dataset_to_mfd(Bhyp, Q, helicoid_mfd)})
# scipy.io.savemat('karate_Qhyp.mat', mdict = {'arr': map_dataset_to_mfd(Bhyp, Q, hyp_mfd)})

#Q =

  # scipy.io.savemat('polblogs_hyp.mat', mdict = {'arr': B})


print ("EUC ORIG ERR: ")
print (err_euc_orig)
print ("EUC LRN ERR: ")
print (err_euc_qlrn)
print ("MFD ORIG ERR: ")
print (err_mfd_orig)
print ("MFD LRN ERR: ")
print (err_mfd_qlrn)


scipy.io.savemat(datasetname+'_clf_err_euc_orig.mat', mdict = {'arr': err_euc_orig})
scipy.io.savemat(datasetname+'_clf_err_euc_qlrn.mat', mdict = {'arr': err_euc_qlrn})
scipy.io.savemat(datasetname+'_clf_err_mfd_orig.mat', mdict = {'arr': err_mfd_orig})
scipy.io.savemat(datasetname+'_clf_err_mfd_qlrn.mat', mdict = {'arr': err_mfd_qlrn})


# ----------------------------------------------------------------------------------------------------
#
# DEPRECATED TESTS BELOW
#
# ----------------------------------------------------------------------------------------------------


    ########     !!!!!!!!    COMMMENT OUT EVERYTHING BELOW !!!!!!!!!!!!!    ##################



# print(discrete_metric)
# B0 = mds_initialization(max_size, DIMENSION)
#
# mds_Powell = minimize(mds_loss, B0, args=(max_size, DIMENSION, discrete_metric, euclid_mfd, euclid_mfd_dist), method='Powell', options={'disp': True})
# print(mds_Powell)
# Bnew = mds_Powell.x.reshape(max_size, DIMENSION)

# Bnew = [ 3.37889783,  0.08641009,  4.03466906,  0.91689482,  2.01794846,
#        0.85489373,  4.67956457,  1.3629206 ,  7.1019306 , -1.47017666,
#        7.90281666, -1.19068525,  8.09079113, -1.33309995,  5.27157478,
#        1.89893081,  1.97554528,  1.28534465,  2.78694838,  2.47883715,
#        9.20954276, -0.52678939, 10.81249127,  0.88102833, 10.49919753,
#        1.06822317,  2.12085145,  0.63625682,  1.60567772,  3.8255304 ,
#        1.60260633,  3.79855365, 23.02476214, -3.11915807,  9.81967872,
#        0.1224998 , -0.05813067,  2.95937075,  1.72823746, -0.30825014,
#        0.56186356,  3.41480805, 10.12324174,  1.93312885, -0.62548791,
#        2.03350795, -0.70267319,  1.36315131,  0.91742442, -1.82649377,
#        0.35692144, -1.53474547, -1.18274845,  0.39341533, -0.09894901,
#       -0.07771349, -0.03397873, -0.10443997, -0.96138015,  0.88029606,
#        1.9222306 ,  1.99712694,  0.85090881, -0.38443464,  0.51403067,
#        1.26898486,  0.29549035,  0.88938318]
# Bnew = np.asarray(Bnew).reshape(max_size, DIMENSION)
# Bnew = scipy.io.loadmat('./karate_hmds.mat')['arr']   ########## <<<<<
# Bnew = [np.asarray(i[1:]) for i in Bnew]
# Labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  #<<<<<<<< GOOD

# >>>     0  0  0  0  0  0  0  0  0  1  0  0  0  0  1  1  0  0  1  0  1  0   1     1     1     1     1     1     1     1     1     1     1     1


###########################################################################################
###########################################################################################
###########################################################################################


# Q0 = np.diag([1 for _ in range(DIMENSION)])
# print("INITIAL EMBEDDING")
# print(map_dataset_to_mfd(Bnew, Q0, hyp_mfd))
# Q0 = minkowski_metric_tensor

# res_NelderMead = minimize(Loss2, Q0, method='nelder-mead', options={'xtol': 1e-3, 'disp': True})
# print(res_NelderMead)

# res_BFGS = minimize(lmnn_Loss, Q0, args=(100, 1, 0.01, deuc), method='BFGS', options={'disp': True})
# print(res_BFGS)

# res_Powell = minimize(lmnn_loss_generic, Q0, args=(100, 2, 0.5, euclid_mfd, euclid_mfd_dist, B, Labels), method='Powell', options={'disp': True})
# print(res_Powell)

#sq_euclid_mfd_dist = lambda x, y : np.linalg.norm(x - y) ** 2
# Centers = kmeans_generic(map_dataset_to_mfd(B, Q0, euclid_mfd), 3, sq_euclid_mfd_dist)

# res_Powell = minimize(mmc_loss_generic, Q0, args=(0.5, hyp_mfd, hyp_mfd_dist, Bnew, Labels), method='Powell', options={'disp': True})
# Qnew = res_Powell.x.reshape(DIMENSION, DIMENSION)
# print(res_Powell)
# print(np.matmul(Qnew.T, Qnew))
#
# print("TRANSFORM")
# print(np.asarray(map_dataset_to_mfd(Bnew, Qnew, hyp_mfd)).tolist())

# ----------------------------------------------------------------------------------------------------
#
# PLOT DATA
#
# ----------------------------------------------------------------------------------------------------

# Initial_Data = B


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
