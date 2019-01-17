import numpy as np
from mfd_functions import map_dataset_to_mfd

def sv_constraint(Q):
    u, s, vh = np.linalg.svd(Q)
    s = s.tolist()
    return max(s)

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

def mmc_loss_generic(Q, reg, lmbd, mfd_generic, mfd_dist_generic, mfd_integrand, B, labels):
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

    total += lmbd * (np.multiply(Q, Q).sum())  ## FINE
    return total
