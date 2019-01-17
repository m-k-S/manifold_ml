import numpy as np
import scipy.io
from config import username, api_key

# ----------------------------------------------------------------------------------------------------
#
# LEARNING M
#
# ----------------------------------------------------------------------------------------------------

from scipy.optimize import minimize
from scipy.integrate import quad
from mfd_functions import *
from metric_loss_functions import *

def get_all_neighbors_of(FQx, label_of_FQx, FQB, labels, radius, k, mfd_dist_generic, mfd_integrand, todebug=False):
################ CODE FOR K
    dst_from_FQx = []
    for idx, FQxi in enumerate(FQB):
        dst_from_FQx.append( mfd_dist_generic(FQx, FQxi, mfd_integrand))

    idx_of_points = np.argsort(np.asarray(dst_from_FQx))  # <<< only need these indices

    true_neighbors = []
    imposter_neighbors = []
    true_neighbors_idx = []
    imposter_neighbors_idx = []

    for i in range(1,k+1):
        nidx = idx_of_points[i]      # +1 because ignoring the zero'th index that is supposed to be FQx itself
        label_nxi = labels[nidx]     # this assumes k < length(FQB)
        if label_nxi == label_of_FQx:
            true_neighbors.append(FQB[nidx])
            true_neighbors_idx.append(nidx)
        else:
            imposter_neighbors.append(FQB[nidx])
            imposter_neighbors_idx.append(nidx)

    # if todebug:
    #     print('>>>> inside compute nbrs')
    #     print(k)
    #     print(dst_from_FQx)
    #     print(idx_of_points)
    #     print(true_neighbors_idx)
    #     print(imposter_neighbors_idx)
    #     print('>>>> call done')
    ##print(true_neighbors)
    ##print(imposter_neighbors)
    ##print('=========================')
    #return true_neighbors, imposter_neighbors
    return true_neighbors_idx, imposter_neighbors_idx

# ----------------------------------------------------------------------------------------------------
#
# LMNN LOSS
#
# ----------------------------------------------------------------------------------------------------

all_FQx_nbrs = {}  #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< THIS IS GLOBAL VAR
all_FQx_impos = {}


def lmnn_loss_generic(Q, radius, k, reg, lmbd, mfd_generic, mfd_dist_generic, mfd_integrand, B, labels, toprintdebug=False):
    dim = len(B[0])
    Q = Q.reshape(dim, dim)
    total = 0
    tmp_v1 = 0
    tmp_v2 = 0

    FQB = map_dataset_to_mfd(B, Q, mfd_generic)
    for idx, FQx in enumerate(FQB):
        label_of_FQx = labels[idx]

        FQy_nbrs_idx, FQz_nbrs_idx = get_all_neighbors_of(FQx, label_of_FQx, FQB, labels, radius, k, mfd_dist_generic, mfd_integrand, toprintdebug)
        #FQy_nbrs, FQz_nbrs = get_all_neighbors_of(B[idx], label_of_FQx, B, labels, radius, k, mfd_dist_generic, mfd_integrand)

        # if toprintdebug:
        #     print('--before try')
        #     print(idx)
        #     print(FQy_nbrs_idx)
        #     print(FQz_nbrs_idx)

        if toprintdebug:
            try:
                all_FQx_nbrs[str(idx)] = list(set(all_FQx_nbrs[str(idx)]).union(set(FQy_nbrs_idx)))
            except KeyError:
                all_FQx_nbrs[str(idx)] = FQy_nbrs_idx
            try: # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  DONT KNOW THE SYNTAX
                all_FQx_impos[str(idx)] = list(set(all_FQx_impos[str(idx)]).union(set(FQz_nbrs_idx)))
            except KeyError:
                all_FQx_impos[str(idx)] = FQz_nbrs_idx

        try:
            tmp_FQx_nbrs = list(set(all_FQx_nbrs[str(idx)]).union(set(FQy_nbrs_idx)))
        except KeyError:
            tmp_FQx_nbrs = FQy_nbrs_idx
        try:
            tmp_FQx_impos =list(set(all_FQx_impos[str(idx)]).union(set(FQz_nbrs_idx)))
        except KeyError:
            tmp_FQx_impos = FQz_nbrs_idx

        # if toprintdebug:
        #     print(idx)
        #     print(all_FQx_nbrs)
        #     print(all_FQx_impos)
        #     print('--end try/catch')

        #for FQy in FQy_nbrs:
        #for FQy_idx in all_FQx_nbrs[str(idx)]:
        for FQy_idx in tmp_FQx_nbrs:
            total += (1 - reg) * mfd_dist_generic(FQx, FQB[FQy_idx], mfd_integrand)
            tmp_v1 += mfd_dist_generic(FQx, FQB[FQy_idx], mfd_integrand)
            #for FQz in FQz_nbrs:
            #for FQz_idx in all_FQx_impos[str(idx)]:
            for FQz_idx in tmp_FQx_impos:
                if mfd_dist_generic(FQx,FQB[FQz_idx], mfd_integrand) < mfd_dist_generic(FQx,FQB[FQy_idx], mfd_integrand)+1:  # +1 is the margin
                    total += reg * (1 + mfd_dist_generic(FQx, FQB[FQy_idx], mfd_integrand) - mfd_dist_generic(FQx, FQB[FQz_idx], mfd_integrand))
                    tmp_v2 += (1 + mfd_dist_generic(FQx, FQB[FQy_idx], mfd_integrand) - mfd_dist_generic(FQx, FQB[FQz_idx], mfd_integrand))

    total += lmbd * (np.multiply(Q, Q).sum())  # YEP

    # if toprintdebug:
    #     print('about to exit lmnn loss call')
    #     print(Q)
    #     print(total)
    #     print(tmp_v1)
    #     print(tmp_v2)
    #     print(all_FQx_nbrs)
    #     print(all_FQx_impos)

    return total

# ----------------------------------------------------------------------------------------------------
#
# PERFORMANCE EVALUATION
#
# ----------------------------------------------------------------------------------------------------

import sklearn.metrics
from learn_dist import integrand_helicoid, learn_distance

def do_classification_tests_all(nrounds, train_ratio, K, reg, lmbd, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
    err_euc_orig = []
    err_euc_qlrn = []
    err_mfd_orig = []
    err_mfd_qlrn = []

    for r in range(nrounds):
        all_FQx_nbrs.clear()
        all_FQx_impos.clear

        eeo,eeq,emo,emq = do_classification_test(train_ratio, K, reg, lmbd, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels)
        err_euc_orig.append(eeo)
        err_euc_qlrn.append(eeq)
        err_mfd_orig.append(emo)
        err_mfd_qlrn.append(emq)

    return err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn


def do_classification_test(train_ratio, K, reg, lmbd, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
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

    # euc_res_Powell = minimize(mmc_loss_generic, Q0_euc, args=(reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True})
    # mfd_res_Powell = minimize(mmc_loss_generic, Q0_mfd, args=(reg, lmbd, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='CG', options={'disp': True})
    # mfd_res_Powell = minimize(mmc_loss_generic, Q0_mfd, args=(reg, lmbd, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='trust-constr', constraints=[sv_cons], options={'disp': True})

    # print('--------------')

    def print_loss(xk):
        loss = lmnn_loss_generic(xk, None, K, reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr, True)
        #print (xk.reshape(dim_euc, dim_euc))
        #print (loss)
        # print('--------------')

    all_FQx_nbrs.clear()  #  <<<<<<<<<<<<<<<<<<
    all_FQx_impos.clear()

    print("EUCLIDEAN INITIAL LOSS: " + str(lmnn_loss_generic(Q0_euc, None, K, reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr, True)))
    euc_res_Powell = minimize(lmnn_loss_generic, Q0_euc, args=(None, K, reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True}, callback=print_loss)
    all_FQx_nbrs.clear()  #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< THIS IS GLOBAL VAR
    all_FQx_impos.clear()

    print("MANIFOLD INITIAL LOSS: " + str(lmnn_loss_generic(Q0_mfd, None, K, reg, lmbd, fxn_mfd, fxn_mfd_dist, None, mfd_data_tr, labels_tr, True)))
    mfd_res_Powell = minimize(lmnn_loss_generic, Q0_mfd, args=(None, K, reg, lmbd, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='Powell', options={'disp': True}, callback=print_loss)
    all_FQx_nbrs.clear()  #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< THIS IS GLOBAL VAR
    all_FQx_impos.clear()
    # mfd_res_Powell = minimize(lmnn_loss_generic, Q0_mfd, args=(100, K, reg, lmbd, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='trust-constr', constraints=[sv_cons], options={'disp': True})
    # print(mfd_res_Powell)


    euc_Qnew = euc_res_Powell.x.reshape(dim_euc, dim_euc)
    mfd_Qnew = mfd_res_Powell.x.reshape(dim_mfd, dim_mfd)

    # print(euc_Qnew)
    # scipy.io.savemat('karate_Qeuc.mat', mdict = {'arr': map_dataset_to_mfd(Bnew_euc, euc_Qnew, euclid_mfd)})

    print (mfd_Qnew)
    # print (np.matmul(mfd_Qnew.T, mfd_Qnew))

    euc_Idata_ts = map_dataset_to_mfd(euc_data_ts, Q0_euc, fxn_euc)
    mfd_Idata_ts = map_dataset_to_mfd(mfd_data_ts, Q0_mfd, fxn_mfd)

    euc_Qdata_ts = map_dataset_to_mfd(euc_data_ts, euc_Qnew, fxn_euc)
    mfd_Qdata_ts = map_dataset_to_mfd(mfd_data_ts, mfd_Qnew, fxn_mfd)

    euc_Qdata_tr = map_dataset_to_mfd(euc_data_tr, euc_Qnew, fxn_euc)
    mfd_Qdata_tr = map_dataset_to_mfd(mfd_data_tr, mfd_Qnew, fxn_mfd)

    euc_Idata_tr = map_dataset_to_mfd(euc_data_tr, Q0_euc, fxn_euc)
    mfd_Idata_tr = map_dataset_to_mfd(mfd_data_tr, Q0_mfd, fxn_mfd)


    euc_lab_tr   = knnclassify_generic(euc_Idata_tr,  K, euc_Idata_tr,  labels_tr, fxn_euc_dist, None, True)
    euc_Qlab_tr  = knnclassify_generic(euc_Qdata_tr,  K, euc_Qdata_tr,  labels_tr, fxn_euc_dist, None, True)
    mfd_lab_tr   = knnclassify_generic(mfd_Idata_tr,  K, mfd_Idata_tr,  labels_tr, fxn_mfd_dist, None, True)
    mfd_Qlab_tr  = knnclassify_generic(mfd_Qdata_tr,  K, mfd_Qdata_tr,  labels_tr, fxn_mfd_dist, None, True)
    err_euc_orig = eval_classification_quality(labels_tr, euc_lab_tr)
    err_euc_qlrn = eval_classification_quality(labels_tr, euc_Qlab_tr)
    err_mfd_orig = eval_classification_quality(labels_tr, mfd_lab_tr)
    err_mfd_qlrn = eval_classification_quality(labels_tr, mfd_Qlab_tr)

    #    # run k-means
    ####K = len(np.unique(true_labels))   # number of unique labels is the value of K in K-means

    #euc_lab_ts   = knnclassify_generic(euc_Idata_ts,  K, euc_Idata_tr,  labels_tr, fxn_euc_dist, None)
    #euc_Qlab_ts  = knnclassify_generic(euc_Qdata_ts, K, euc_Qdata_tr, labels_tr, fxn_euc_dist, None)
    #mfd_lab_ts   = knnclassify_generic(mfd_Idata_ts,  K, mfd_Idata_tr,  labels_tr, fxn_mfd_dist, None)
    #mfd_Qlab_ts  = knnclassify_generic(mfd_Qdata_ts, K, mfd_Qdata_tr, labels_tr, fxn_mfd_dist, None)

        # evaluate classification results
    #err_euc_orig = eval_classification_quality(labels_ts, euc_lab_ts)
    #err_euc_qlrn = eval_classification_quality(labels_ts, euc_Qlab_ts)
    #err_mfd_orig = eval_classification_quality(labels_ts, mfd_lab_ts)
    #err_mfd_qlrn = eval_classification_quality(labels_ts, mfd_Qlab_ts)

    return err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn

def eval_classification_quality(labels_ts, mfd_lab_ts):
    err_01 = 0
    for i in range(len(labels_ts)):
        if labels_ts[i][0] != mfd_lab_ts[i]:
            err_01 += 1

    err_01 /= len(labels_ts)
    return err_01

def knnclassify_generic(data_ts,  K, data_tr, labels_tr, mfd_dist_generic, mfd_integrand, skip_first_nbr=False):
    labels_ts = []

    for x_ts in data_ts:
        x_ts = np.asarray(x_ts)
        dst_to_xts = []
        for idx_tr, x_tr in enumerate(data_tr):
            x_tr = np.asarray(x_tr)
            dst_to_xts.append( mfd_dist_generic(x_tr, x_ts, mfd_integrand))

        idx_of_points = np.argsort(np.asarray(dst_to_xts))  # <<< only need these indices

        l = [];

        if skip_first_nbr:
            for i in idx_of_points[1:K+1]:
                l.append(labels_tr[i])
        else:
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
# datasetname = 'karate'
# datasetname = 'football'
datasetname = 'polbooks'
# datasetname = 'polblogs'

if datasetname == 'karate':
    Beuc   = scipy.io.loadmat('./data/karate_euc.mat')['arr']
    Bhyp   = scipy.io.loadmat('./data/karate_gmds_hyp.mat')['arr'] # this file has correct data in it!
    Labels = scipy.io.loadmat('./data/karate_data_1.mat')['label']
    train_ratio = 0.6
    fxn_mfd = hyp_mfd
    fxn_mfd_dist = hyp_mfd_dist
    #fxn_integrand = integrand_hyp
    fxn_integrand = None

elif datasetname == 'helicoid':
    Beuc = scipy.io.loadmat('./data/helicoid.mat')['data']  # 3 D
    Bhyp   = scipy.io.loadmat('./data/helicoid.mat')['base'] # 2 D
    Labels = scipy.io.loadmat('./data/helicoid.mat')['labels']
    train_ratio = 0.6
    fxn_mfd = helicoid_mfd
    fxn_mfd_dist = helicoid_mfd_dist
    fxn_integrand = integrand_helicoid

elif datasetname == 'football':
    Beuc   = scipy.io.loadmat('./data/football_euc.mat')['arr']
    Bhyp   = scipy.io.loadmat('./data/football_gmds_hyp.mat')['arr'] # this file has correct data in it!
    Labels = scipy.io.loadmat('./data/football_data_1.mat')['label']
    train_ratio = 0.75
    fxn_mfd = hyp_mfd
    fxn_mfd_dist = hyp_mfd_dist
    #fxn_integrand = integrand_hyp
    fxn_integrand = None

elif datasetname == 'polbooks':
    Beuc   = scipy.io.loadmat('./data/polbooks_euc.mat')['arr']
    Bhyp   = scipy.io.loadmat('./data/polbooks_gmds_hyp.mat')['arr'] # this file has correct data in it!
    Labels = scipy.io.loadmat('./data/polbooks_data_1.mat')['label']
    train_ratio = 0.7
    fxn_mfd = hyp_mfd
    fxn_mfd_dist = hyp_mfd_dist
    #fxn_integrand = integrand_hyp
    fxn_integrand = None

elif datasetname == 'polblogs':
    Beuc   = scipy.io.loadmat('./data/polblogs_euc.mat')['arr']
    Bhyp   = scipy.io.loadmat('./data/polblogs_gmds_hyp.mat')['arr']  # FILE DOES NOT EXIST!  CORRESPONDING .txt doesnt exist (because the run was freezing)
    Labels = scipy.io.loadmat('./data/polblogs_data_1.mat')['label']
    train_ratio = 0.8
    fxn_mfd = hyp_mfd
    fxn_mfd_dist = hyp_mfd_dist
    #fxn_integrand = integrand_hyp
    fxn_integrand = None

else:
    print('undefined dataset!')
    assert(1==0)



fxn_euc = euclid_mfd
fxn_euc_dist = euclid_mfd_dist


# err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn = do_cluster_tests_all(nrounds, train_ratio, Beuc, fxn_euc, fxn_euc_dist, Bhyp, fxn_mfd, fxn_mfd_dist, fxn_integrand, Labels)
#
# scipy.io.savemat('helicoid_Q.mat', mdict = {'arr': map_dataset_to_mfd(Bhyp, Q, helicoid_mfd)})
# scipy.io.savemat('karate_Qhyp.mat', mdict = {'arr': map_dataset_to_mfd(Bhyp, Q, hyp_mfd)})

#Q =

  # scipy.io.savemat('polblogs_hyp.mat', mdict = {'arr': B})
nrounds = 10
k = 3
reg = 0.5
lmbd = 0.0
err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn = do_classification_tests_all(nrounds, train_ratio, k, reg, lmbd, Beuc, fxn_euc, fxn_euc_dist, Bhyp, fxn_mfd, fxn_mfd_dist, fxn_integrand, Labels)

scipy.io.savemat(datasetname+'_clf_err_euc_orig_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_euc_orig})
scipy.io.savemat(datasetname+'_clf_err_euc_qlrn_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_euc_qlrn})
scipy.io.savemat(datasetname+'_clf_err_mfd_orig_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_mfd_orig})
scipy.io.savemat(datasetname+'_clf_err_mfd_qlrn_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_mfd_qlrn})
# scipy.io.savemat('../../../../Dropbox/max - hyperbolic_mlearn/results/'+datasetname+'/'+datasetname+'_clf_err_euc_orig_reg'+str(reg)+'_k'+str(k)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_euc_orig})
# scipy.io.savemat('../../../../Dropbox/max - hyperbolic_mlearn/results/'+datasetname+'/'+datasetname+'_clf_err_euc_qlrn_reg'+str(reg)+'_k'+str(k)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_euc_qlrn})
# scipy.io.savemat('../../../../Dropbox/max - hyperbolic_mlearn/results/'+datasetname+'/'+datasetname+'_clf_err_mfd_orig_reg'+str(reg)+'_k'+str(k)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_mfd_orig})
# scipy.io.savemat('../../../../Dropbox/max - hyperbolic_mlearn/results/'+datasetname+'/'+datasetname+'_clf_err_mfd_qlrn_reg'+str(reg)+'_k'+str(k)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_mfd_qlrn})

'''
for datasetname in ['helicoid', 'polbooks', 'football']:
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
        train_ratio = 0.75
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

    for k in [1, 2, 3, 5, 8, 12]:
        for reg in [0.1, 0.2, 0.4, 0.5, 0.7, 0.8]:
            err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn = do_classification_tests_all(nrounds, train_ratio, k, reg, Beuc, fxn_euc, fxn_euc_dist, Bhyp, fxn_mfd, fxn_mfd_dist, fxn_integrand, Labels)
            scipy.io.savemat(datasetname+'_clf_err_euc_orig_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_euc_orig})
            scipy.io.savemat(datasetname+'_clf_err_euc_qlrn_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_euc_qlrn})
            scipy.io.savemat(datasetname+'_clf_err_mfd_orig_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_mfd_orig})
            scipy.io.savemat(datasetname+'_clf_err_mfd_qlrn_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_mfd_qlrn})

            scipy.io.savemat('../../../../Dropbox/max - hyperbolic_mlearn/results/'+datasetname+'/'+datasetname+'_clf_err_euc_orig_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_euc_orig})
            scipy.io.savemat('../../../../Dropbox/max - hyperbolic_mlearn/results/'+datasetname+'/'+datasetname+'_clf_err_euc_qlrn_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_euc_qlrn})
            scipy.io.savemat('../../../../Dropbox/max - hyperbolic_mlearn/results/'+datasetname+'/'+datasetname+'_clf_err_mfd_orig_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_mfd_orig})
            scipy.io.savemat('../../../../Dropbox/max - hyperbolic_mlearn/results/'+datasetname+'/'+datasetname+'_clf_err_mfd_qlrn_reg'+str(reg)+'_k'+str(k)+'.mat', mdict = {'arr': err_mfd_qlrn})
'''

# print ("EUC ORIG ERR: ")
# print (err_euc_orig)
# print ("EUC LRN ERR: ")
# print (err_euc_qlrn)
# print ("MFD ORIG ERR: ")
# print (err_mfd_orig)
# print ("MFD LRN ERR: ")
# print (err_mfd_qlrn)
#
