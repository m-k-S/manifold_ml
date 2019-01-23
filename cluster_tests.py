import numpy as np
from loss_functions import *
from manifold_functions import map_dataset_to_mfd
from scipy.optimize import minimize
import random
import sklearn.metrics

def kmeans_randomly_partition_data(FQB, k):
    assigned_labels = []
    for _ in FQB:
        assigned_labels.append(random.choice(range(k)))
    return assigned_labels

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


def do_cluster_test(train_ratio, k, reg, lmbd, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
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
        if np.random.random() < train_ratio:
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

    euc_res_Powell = minimize(mmc_loss_generic, Q0_euc, args=(reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True})
    mfd_res_Powell = minimize(mmc_loss_generic, Q0_mfd, args=(reg, lmbd, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='Powell', options={'disp': True})

    euc_Qnew = euc_res_Powell.x.reshape(dim_euc, dim_euc)
    mfd_Qnew = mfd_res_Powell.x.reshape(dim_mfd, dim_mfd)

    euc_Qdata_ts = map_dataset_to_mfd(euc_data_ts, euc_Qnew, fxn_euc)
    mfd_Qdata_ts = map_dataset_to_mfd(mfd_data_ts, mfd_Qnew, fxn_mfd)
    euc_Idata_ts = map_dataset_to_mfd(euc_data_ts, Q0_euc, fxn_euc)
    mfd_Idata_ts = map_dataset_to_mfd(mfd_data_ts, Q0_mfd, fxn_mfd)

        # run k-means
    K = len(np.unique(true_labels))   # number of unique labels is the value of K in K-means

    euc_lab_ts  = kmeans_generic(euc_Idata_ts, K, fxn_euc_dist, None)
    euc_Qlab_ts = kmeans_generic(euc_Qdata_ts, K, fxn_euc_dist, None)
    mfd_lab_ts  = kmeans_generic(mfd_Idata_ts, K, fxn_mfd_dist, fxn_integrand)
    mfd_Qlab_ts = kmeans_generic(mfd_Qdata_ts, K, fxn_mfd_dist, fxn_integrand)

        # evaluate k-means results
    err_euc_orig = eval_cluster_quality(labels_ts, euc_lab_ts)
    err_euc_qlrn = eval_cluster_quality(labels_ts, euc_Qlab_ts)
    err_mfd_orig = eval_cluster_quality(labels_ts, mfd_lab_ts)
    err_mfd_qlrn = eval_cluster_quality(labels_ts, mfd_Qlab_ts)

    return err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn


def eval_cluster_quality(true_labels, assigned_labels):
    true_labels = [i[0] for i in true_labels]

    ARI = sklearn.metrics.adjusted_rand_score(true_labels, assigned_labels)
    NMI = sklearn.metrics.normalized_mutual_info_score(true_labels, assigned_labels)
    err = [ARI, NMI]
    return err

def do_cluster_tests_all(nrounds, train_ratio, k, reg, lmbd, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
    err_euc_orig = []
    err_euc_qlrn = []
    err_mfd_orig = []
    err_mfd_qlrn = []

    for r in range(nrounds):
        eeo,eeq,emo,emq = do_cluster_test(train_ratio, k, reg, lmbd, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels)
        err_euc_orig.append(eeo)
        err_euc_qlrn.append(eeq)
        err_mfd_orig.append(emo)
        err_mfd_qlrn.append(emq)

    return err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn
