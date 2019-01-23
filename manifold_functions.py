# ----------------------------------------------------------------------------------------------------
#
# AUXILIARY FUNCTIONS FOR COMPUTATIONS ON SPECIFIED MANIFOLDS
#
# ----------------------------------------------------------------------------------------------------

import numpy as np
from learn_manifold_distance import learn_distance

# FUNCTIONS FOR HELICOID MANIFOLD:

# Input: base space coordinates
# Output: helicoid coordinates
def helicoid_mfd(x):
    r = x[0]
    t = x[1]
    mp = [r*np.cos(t), r*np.sin(t), t]
    return mp   # helicoid map

# Arc length integrand for the helicoid; used in computing distance between points on the helicoid
def integrand_helicoid(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x   # 2 x 1

    r = Pth[0]
    s = Pth[1]
    D = [[np.cos(s), -r * np.sin(s)],
        [np.sin(s), r * np.cos(s)],
        [0, 1]]

    v = np.matmul(D, np.matmul(Q, Dff))
    return np.linalg.norm(v)

# Input: x,y points on helicoid
# Output: apprxoimate distance between x,y
def helicoid_mfd_dist(x, y, integrand=integrand_helicoid):
    xr = x[0] / np.cos(x[2])
    xs = x[2]

    yr = y[0]/np.cos(y[2])
    ys = y[2]

    bx = np.asarray([np.abs(xr), xs])
    by = np.asarray([np.abs(yr), ys])

    dist = learn_distance(bx, by, I, integrand)
    return dist

# FUNCTIONS FOR HYPERBOLOID:

# Input: base space coordinates
# Output: hyperboloid coordinates
def hyp_mfd(x):
    x0 = np.sqrt(1 + np.sum(np.power(x, 2)))
    x = np.concatenate(([x0], x))
    return x   # hyperbolic map

# Input: x,y points on hyperboloid
# Output: distance between x,y on hyperboloid
def hyp_mfd_dist(x,y, integrand):
    x0 = x[0]
    y0 = y[0]
    xtail = x[1:]
    ytail = y[1:]

    xGy = np.matmul(xtail.T,ytail) - x0*y0
    return np.arccosh(-xGy)

# FUNCTIONS FOR EUCLIDEAN SPACE:

# Base space is equal to manifold, thus identity function
def euclid_mfd(x):
    return x   # identity map

# Euclidean distance
def euclid_mfd_dist(x,y, integrand):
    return np.linalg.norm(x-y)


# This function maps a dataset in the base Euclidean space (modified by a matrix Q)
# onto a specified manifold
def map_dataset_to_mfd(B, Q, mfd_generic):
    FQB = []
    for x in B:
        Qx = np.matmul(Q, x)
        mQx = mfd_generic(Qx)
        FQB.append(mQx)

    return FQB