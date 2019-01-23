import numpy as np
import scipy.io

# ----------------------------------------------------------------------------------------------------
#
# PERFORM TESTS
#
# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    from cluster_tests import *
    from classification_tests import *
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--K')
    parser.add_argument('--reg')
    parser.add_argument('--lmbd')
    parser.add_argument('--clf', action='store_true')
    parser.add_argument('--clus', action='store_true')
    args = parser.parse_args()

    k = int(args.K)
    reg = float(args.reg)
    lmbd = float(args.lmbd)
    datasetname = args.dataset

    if datasetname == 'karate':
        Beuc   = scipy.io.loadmat('./data/karate.mat')['Beuc']
        Bhyp   = scipy.io.loadmat('./data/karate.mat')['Bhyp']
        Labels = scipy.io.loadmat('./data/karate.mat')['Labels']
        train_ratio = 0.8
        fxn_mfd = hyp_mfd
        fxn_mfd_dist = hyp_mfd_dist
        fxn_integrand = None

    elif datasetname == 'helicoid':
        Beuc = scipy.io.loadmat('./data/helicoid.mat')['data']  # 3D
        Bhyp   = scipy.io.loadmat('./data/helicoid.mat')['base'] # 2D
        Labels = scipy.io.loadmat('./data/helicoid.mat')['labels']
        train_ratio = 0.6
        fxn_mfd = helicoid_mfd
        fxn_mfd_dist = helicoid_mfd_dist
        fxn_integrand = integrand_helicoid

    elif datasetname == 'football':
        Beuc   = scipy.io.loadmat('./data/football.mat')['Beuc']
        Bhyp   = scipy.io.loadmat('./data/football.mat')['Bhyp']
        Labels = scipy.io.loadmat('./data/football.mat')['Labels']
        train_ratio = 0.75
        fxn_mfd = hyp_mfd
        fxn_mfd_dist = hyp_mfd_dist
        fxn_integrand = None

    elif datasetname == 'polbooks':
        Beuc   = scipy.io.loadmat('./data/polbooks.mat')['Beuc']
        Bhyp   = scipy.io.loadmat('./data/polbooks.mat')['Bhyp']
        Labels = scipy.io.loadmat('./data/polbooks.mat')['Labels']
        train_ratio = 0.7
        fxn_mfd = hyp_mfd
        fxn_mfd_dist = hyp_mfd_dist
        fxn_integrand = None

    elif datasetname == 'adjnoun':
        Beuc = scipy.io.loadmat('./data/adjnoun.mat')['Beuc']
        Bhyp = scipy.io.loadmat('./data/adjnoun.mat')['Bhyp']
        Labels = scipy.io.loadmat('./data/adjnoun.mat')['Labels']
        train_ratio = 0.7
        fxn_mfd = hyp_mfd
        fxn_mfd_dist = hyp_mfd_dist
        fxn_integrand = None

    elif datasetname == '20newsgroup':
        Beuc = scipy.io.loadmat('./data/20newsgroup.mat')['Beuc']
        Bhyp = scipy.io.loadmat('./data/20newsgroup.mat')['Bhyp']
        Labels = scipy.io.loadmat('./data/20newsgroup.mat')['Labels']
        train_ratio = 0.7
        fxn_mfd = hyp_mfd
        fxn_mfd_dist = hyp_mfd_dist
        fxn_integrand = None

    else:
        print('Undefined dataset!')
        exit()

    fxn_euc = euclid_mfd
    fxn_euc_dist = euclid_mfd_dist
    nrounds = 10

    if args.clf:
        err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn = do_classification_tests_all(nrounds, train_ratio, k, reg, lmbd, Beuc, fxn_euc, fxn_euc_dist, Bhyp, fxn_mfd, fxn_mfd_dist, fxn_integrand, Labels)
        scipy.io.savemat('./'+datasetname+'/'+datasetname+'_CLF_err_euc_orig_reg'+str(reg)+'_k'+str(k)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_euc_orig})
        scipy.io.savemat('./'+datasetname+'/'+datasetname+'_CLF_err_euc_qlrn_reg'+str(reg)+'_k'+str(k)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_euc_qlrn})
        scipy.io.savemat('./'+datasetname+'/'+datasetname+'_CLF_err_mfd_orig_reg'+str(reg)+'_k'+str(k)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_mfd_orig})
        scipy.io.savemat('./'+datasetname+'/'+datasetname+'_CLF_err_mfd_qlrn_reg'+str(reg)+'_k'+str(k)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_mfd_qlrn})

    elif args.clus:
        err_euc_orig, err_euc_qlrn, err_mfd_orig, err_mfd_qlrn = do_cluster_tests_all(nrounds, train_ratio, k, reg, lmbd, Beuc, fxn_euc, fxn_euc_dist, Bhyp, fxn_mfd, fxn_mfd_dist, fxn_integrand, Labels)
        scipy.io.savemat('./'+datasetname+'/'+datasetname+'_CLUS_err_euc_orig_reg'+str(reg)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_euc_orig})
        scipy.io.savemat('./'+datasetname+'/'+datasetname+'_CLUS_err_euc_qlrn_reg'+str(reg)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_euc_qlrn})
        scipy.io.savemat('./'+datasetname+'/'+datasetname+'_CLUS_err_mfd_orig_reg'+str(reg)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_mfd_orig})
        scipy.io.savemat('./'+datasetname+'/'+datasetname+'_CLUS_err_mfd_qlrn_reg'+str(reg)+'_lmbd'+str(lmbd)+'.mat', mdict = {'arr': err_mfd_qlrn})

    else:
        print("No test type specified, exiting.")
        exit()
