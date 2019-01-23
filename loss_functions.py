import numpy as np
from manifold_functions import map_dataset_to_mfd

# ----------------------------------------------------------------------------------------------------
#
# MMC LOSS FUNCTION
#
# ----------------------------------------------------------------------------------------------------

# Gets all similarity and dissimilarity pairs in a dataset (for MMC loss function)
# Input: labels of points
# Output: pairs of indices of similar points
#         pairs of indices of dissimilar points
def get_sim_dis_pairs(labels):
    sim_idxs = []
    dis_idxs = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                sim_idxs.append( [i, j] )
            else:
                dis_idxs.append( [i, j] )

    return sim_idxs, dis_idxs

# MMC Loss Function
# Inputs:
#   Q - variable of optimization, the linear transformation affecting the dataset
#   reg - regularization term
#   lmbd - term to penalize large scaling of data
#   mfd_generic - maps points in the base space to the manifold
#   mfd_dist_generic - computes distances between points on the manifold
#   mfd_integrand - integrand of the arc length integral of the manifold, if explicit distance is unknown
#   B - dataset of points in base space
#   labels - labels of points
# Outputs:
#   loss - value of loss for given parameters
def mmc_loss_generic(Q, reg, lmbd, mfd_generic, mfd_dist_generic, mfd_integrand, B, labels):
    dim = len(B[0])
    Q = Q.reshape(dim, dim)

    total = 0
    FQB = map_dataset_to_mfd(B, Q, mfd_generic)
    sim_idxs, dis_idxs = get_sim_dis_pairs(labels)

    nsim_idxs = len(sim_idxs)
    ndis_idxs = len(dis_idxs)

    for sim_idx in sim_idxs:
        total += (1-reg) * mfd_dist_generic(FQB[sim_idx[0]], FQB[sim_idx[1]], mfd_integrand) / nsim_idxs

    for dis_idx in dis_idxs:
        total -= (reg)   * mfd_dist_generic(FQB[dis_idx[0]], FQB[dis_idx[1]], mfd_integrand) / ndis_idxs

    total += lmbd * (np.multiply(Q, Q).sum())
    return total

# ----------------------------------------------------------------------------------------------------
#
# LMNN LOSS FUNCTION
#
# ----------------------------------------------------------------------------------------------------

# Returns indices of true neighbors (neighbors with same label) and imposter neighbors (neighbors with different label)
def get_all_neighbors_of(FQx, label_of_FQx, FQB, labels, radius, k, mfd_dist_generic, mfd_integrand, update=False):
    dst_from_FQx = []
    for idx, FQxi in enumerate(FQB):
        dst_from_FQx.append( mfd_dist_generic(FQx, FQxi, mfd_integrand))

    idx_of_points = np.argsort(np.asarray(dst_from_FQx))

    true_neighbors = []
    imposter_neighbors = []
    true_neighbors_idx = []
    imposter_neighbors_idx = []

    for i in range(1,k+1):
        nidx = idx_of_points[i]
        label_nxi = labels[nidx]
        if label_nxi == label_of_FQx:
            true_neighbors.append(FQB[nidx])
            true_neighbors_idx.append(nidx)
        else:
            imposter_neighbors.append(FQB[nidx])
            imposter_neighbors_idx.append(nidx)

    return true_neighbors_idx, imposter_neighbors_idx

all_FQx_nbrs = {}
all_FQx_impos = {}

def lmnn_loss_generic(Q, radius, k, reg, lmbd, mfd_generic, mfd_dist_generic, mfd_integrand, B, labels, update=False):
    dim = len(B[0])
    Q = Q.reshape(dim, dim)
    total = 0
    tmp_v1 = 0
    tmp_v2 = 0

    FQB = map_dataset_to_mfd(B, Q, mfd_generic)
    for idx, FQx in enumerate(FQB):
        label_of_FQx = labels[idx]

        if update:
            FQy_nbrs_idx, FQz_nbrs_idx = get_all_neighbors_of(FQx, label_of_FQx, FQB, labels, radius, k, mfd_dist_generic, mfd_integrand, update)
            try:
                all_FQx_nbrs[str(idx)] = list(set(all_FQx_nbrs[str(idx)]).union(set(FQy_nbrs_idx)))
            except KeyError:
                all_FQx_nbrs[str(idx)] = FQy_nbrs_idx
            try:
                all_FQx_impos[str(idx)] = list(set(all_FQx_impos[str(idx)]).union(set(FQz_nbrs_idx)))
            except KeyError:
                all_FQx_impos[str(idx)] = FQz_nbrs_idx
        else:
            FQy_nbrs_idx = []
            FQz_nbrs_idx = []

        try:
            tmp_FQx_nbrs = list(set(all_FQx_nbrs[str(idx)]).union(set(FQy_nbrs_idx)))
        except KeyError:
            tmp_FQx_nbrs = FQy_nbrs_idx
        try:
            tmp_FQx_impos =list(set(all_FQx_impos[str(idx)]).union(set(FQz_nbrs_idx)))
        except KeyError:
            tmp_FQx_impos = FQz_nbrs_idx

        for FQy_idx in tmp_FQx_nbrs:
            total += (1 - reg) * mfd_dist_generic(FQx, FQB[FQy_idx], mfd_integrand)
            tmp_v1 += mfd_dist_generic(FQx, FQB[FQy_idx], mfd_integrand)

            for FQz_idx in tmp_FQx_impos:
                if mfd_dist_generic(FQx,FQB[FQz_idx], mfd_integrand) < mfd_dist_generic(FQx,FQB[FQy_idx], mfd_integrand)+1:  # +1 is the margin
                    total += reg * (1 + mfd_dist_generic(FQx, FQB[FQy_idx], mfd_integrand) - mfd_dist_generic(FQx, FQB[FQz_idx], mfd_integrand))
                    tmp_v2 += (1 + mfd_dist_generic(FQx, FQB[FQy_idx], mfd_integrand) - mfd_dist_generic(FQx, FQB[FQz_idx], mfd_integrand))

    total += lmbd * (np.multiply(Q, Q).sum())

    return total
