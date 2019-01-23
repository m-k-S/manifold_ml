import numpy as np
from loss_functions import *
from manifold_functions import map_dataset_to_mfd
from scipy.optimize import minimize
import scipy.stats

def do_classification_tests_all(nrounds, train_ratio, K, reg, lmbd, Bnew_euc, fxn_euc, fxn_euc_dist, Bnew_mfd, fxn_mfd, fxn_mfd_dist, fxn_integrand, true_labels):
    err_euc_orig = []
    err_euc_qlrn = []
    err_mfd_orig = []
    err_mfd_qlrn = []

    for r in range(nrounds):
        all_FQx_nbrs.clear()
        all_FQx_impos.clear()

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

    all_FQx_nbrs.clear()
    all_FQx_impos.clear()

    print("EUCLIDEAN INITIAL LOSS: " + str(lmnn_loss_generic(Q0_euc, None, K, reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr, True)))
    euc_res_Powell = minimize(lmnn_loss_generic, Q0_euc, args=(None, K, reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True})
    for count in range(5):
        print("EUCLIDEAN IN PROGRESS LOSS: " + str(lmnn_loss_generic(euc_res_Powell.x, None, K, reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr, True)))
        euc_res_Powell = minimize(lmnn_loss_generic, euc_res_Powell.x, args=(None, K, reg, lmbd, fxn_euc, fxn_euc_dist, None, euc_data_tr, labels_tr), method='Powell', options={'disp': True})

    all_FQx_nbrs.clear()
    all_FQx_impos.clear()

    print("MANIFOLD INITIAL LOSS: " + str(lmnn_loss_generic(Q0_mfd, None, K, reg, lmbd, fxn_mfd, fxn_mfd_dist, None, mfd_data_tr, labels_tr, True)))
    mfd_res_Powell = minimize(lmnn_loss_generic, Q0_mfd, args=(None, K, reg, lmbd, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='Powell', options={'disp': True})
    for count in range(5):
        print("MANIFOLD IN PROGRESS LOSS: " + str(lmnn_loss_generic(mfd_res_Powell.x, None, K, reg, lmbd, fxn_mfd, fxn_mfd_dist, None, mfd_data_tr, labels_tr, True)))
        mfd_res_Powell = minimize(lmnn_loss_generic, mfd_res_Powell.x, args=(None, K, reg, lmbd, fxn_mfd, fxn_mfd_dist, fxn_integrand, mfd_data_tr, labels_tr), method='Powell', options={'disp': True})

    all_FQx_nbrs.clear()
    all_FQx_impos.clear()

    euc_Qnew = euc_res_Powell.x.reshape(dim_euc, dim_euc)
    mfd_Qnew = mfd_res_Powell.x.reshape(dim_mfd, dim_mfd)

    euc_Idata_ts = map_dataset_to_mfd(euc_data_ts, Q0_euc, fxn_euc)
    mfd_Idata_ts = map_dataset_to_mfd(mfd_data_ts, Q0_mfd, fxn_mfd)

    euc_Qdata_ts = map_dataset_to_mfd(euc_data_ts, euc_Qnew, fxn_euc)
    mfd_Qdata_ts = map_dataset_to_mfd(mfd_data_ts, mfd_Qnew, fxn_mfd)

    euc_Qdata_tr = map_dataset_to_mfd(euc_data_tr, euc_Qnew, fxn_euc)
    mfd_Qdata_tr = map_dataset_to_mfd(mfd_data_tr, mfd_Qnew, fxn_mfd)

    euc_Idata_tr = map_dataset_to_mfd(euc_data_tr, Q0_euc, fxn_euc)
    mfd_Idata_tr = map_dataset_to_mfd(mfd_data_tr, Q0_mfd, fxn_mfd)

    euc_lab_ts   = knnclassify_generic(euc_Idata_ts,  K, euc_Idata_tr,  labels_tr, fxn_euc_dist, None, False)
    euc_Qlab_ts  = knnclassify_generic(euc_Qdata_ts,  K, euc_Qdata_tr,  labels_tr, fxn_euc_dist, None, False)
    mfd_lab_ts   = knnclassify_generic(mfd_Idata_ts,  K, mfd_Idata_tr,  labels_tr, fxn_mfd_dist, None, False)
    mfd_Qlab_ts  = knnclassify_generic(mfd_Qdata_ts,  K, mfd_Qdata_tr,  labels_tr, fxn_mfd_dist, None, False)

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

def knnclassify_generic(data_ts,  K, data_tr, labels_tr, mfd_dist_generic, mfd_integrand, skip_first_nbr=False):
    labels_ts = []

    for x_ts in data_ts:
        x_ts = np.asarray(x_ts)
        dst_to_xts = []
        for idx_tr, x_tr in enumerate(data_tr):
            x_tr = np.asarray(x_tr)
            dst_to_xts.append( mfd_dist_generic(x_tr, x_ts, mfd_integrand))

        idx_of_points = np.argsort(np.asarray(dst_to_xts))

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
