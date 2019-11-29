# ----------------------------------------------------------------------------------------------------
#
# AUXILIARY FUNCTIONS FOR COMPUTATIONS ON SPECIFIED MANIFOLDS
#
# ----------------------------------------------------------------------------------------------------

import numpy as np
from learn_manifold_distance import learn_distance
from scipy.optimize import minimize


# FUNCTIONS FOR KLEIN BOTTLE:
def klein_mfd(x):
    u = x[0]
    v = x[1]
    mp = [np.cos(u) * (np.cos(0.5 * u) * (np.sqrt(2) + np.cos(v)) + np.sin(0.5 * u) * np.sin(v) * np.cos(v)), \
          np.sin(u) * (np.cos(0.5 * u) * (np.sqrt(2) + np.cos(v)) + np.sin(0.5 * u) * np.sin(v) * np.cos(v)), \
          -np.sin(0.5 * u) * (np.sqrt(2) + np.cos(v)) + np.cos(0.5 * u) * np.sin(v) * np.cos(v)]
    return mp

def integrand_klein(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x   # 2 x 1
    u = Pth[0]
    v = Pth[1]

    a = np.cos(0.5 * u) * (np.sqrt(2) + np.cos(v)) + np.sin(0.5 * u) * np.sin(v) * np.cos(v)
    da_du = -0.5 * np.sin(0.5 * u) * (np.sqrt(2) + np.cos(v)) + 0.5 * np.cos(0.5 * u) * np.sin(v) * np.cos(v)
    da_dv = -np.sin(v) * np.cos(0.5 * u) + np.sin(0.5 * u) * (np.cos(2 * v))

    D = [ [np.cos(u) * da_du - a * np.sin(u), np.cos(u) * da_dv], \
          [np.sin(u) * da_du + a * np.cos(u), np.sin(u) * da_dv], \
          [-0.5 * np.cos(0.5 * u) * (np.sqrt(2) + np.cos(v)) - 0.5 * np.sin(0.5 * u) * np.sin(v) * np.cos(v), \
                            np.sin(v) * np.sin(0.5 * u) + np.cos(0.5 * u) (np.cos(2 * v))]]

    v = np.matmul(D, np.matmul(Q, Dff))
    return np.linalg.norm(v)

def klein_obj(v, u, z):
    return (-np.sin(0.5 * u) * (np.sqrt(2) + np.cos(v)) + 0.5 * np.cos(0.5 * 2) * np.sin(2 * v) - z)**2

def klein_mfd_dist(x, y, integrand=integrand_klein):
    xu = np.arctan(x[1]/x[0])
    xv0 = 2
    min_coord = minimize(klein_obj, xv0, args=(xu, x[2]), method='Powell', options={'disp': True})
    xv = min_coord.x

    yu = np.arctan(y[1]/y[0])
    yv0 = 2
    min_coord = minimize(klein_obj, yv0, args=(yu, y[2]), method='Powell', options={'disp': True})
    yv = min_coord.x

    bx = np.asarray([np.abs(xu), xv])
    by = np.asarray([np.abs(yu), yv])
    I = np.identity(2)
    dist = learn_distance(bx, by, I, integrand)
    return dist


# FUNCTIONS FOR SWISSROLL MANIFOLD:
def swiss_mfd(x):
    r = x[0]
    s = x[1]
    mp = [r * np.cos(r), s, r * np.sin(r)]
    return mp

def integrand_swiss(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x   # 2 x 1

    r = Pth[0]
    s = Pth[1]
    D = np.asarray([[np.cos(r) - r * np.sin(r), 0],
        [0, 1],
        [np.sin(r) + r * np.cos(r), 0]]).astype(float)
    v = np.matmul(D, np.matmul(Q, Dff))
    return np.linalg.norm(v)

def swiss_mfd_dist(x, y, integrand=integrand_swiss):
    xr = np.arctan(x[2] / x[0])
    xs = x[1]

    yr = np.arctan(y[2] / y[0])
    ys = y[1]

    bx = np.asarray([np.abs(xr), xs])
    by = np.asarray([np.abs(yr), ys])
    dist = learn_distance(bx, by, np.eye(2), integrand)
    return dist

# FUNCTIONS FOR TORUS MANIFOLD
# Outer radius 4, inner radius 1

def torus_mfd(x):
    r = x[0]
    s = x[1]
    mp = [4 + np.cos(r) * np.cos(s), 4 + np.cos(r) * np.sin(s), np.sin(r)]
    return mp

def integrand_torus(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x   # 2 x 1

    r = Pth[0]
    s = Pth[1]
    D = np.asarray([[-np.sin(r) * np.cos(s), -np.sin(s) * np.cos(r)],
        [-np.sin(r) * np.sin(s), np.cos(r) * np.cos(s)],
        [np.cos(r), 0]])
    v = np.matmul(D, np.matmul(Q, Dff))
    return np.linalg.norm(v)

def torus_mfd_dist(x, y, integrand=integrand_torus):
    xr = np.arcsin(x[2])
    xs = np.arctan( (x[1] - 4) / (x[0] - 4))

    yr = np.arcsin(y[2])
    ys = np.arctan( (y[1] - 4) / (y[0] - 4))

    bx = np.asarray([np.abs(xr), xs])
    by = np.asarray([np.abs(yr), ys])

    dist = learn_distance(bx, by, np.eye(2), integrand)
    return dist

# FUNCTIONS FOR TREFOIL MANIFOLD

def trefoil_mfd(x):
    return [np.cos(x) + 2 * np.cos(2 * x), np.sin(x) - 2 * np.sin(2 * x), 2 * np.sin(3*x)]

def trefoil_mfd_base_dist(x, y):
    xt = np.arcsin(x[2] / 2) / 3
    yt = np.arcsin(y[2] / 2) / 3

    return np.linalg.norm(xt - yt)

def integrand_trefoil(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x   # 1 x 1

    r = Pth[0]
    D = np.asarray([[-np.sin(r) - 4 * np.sin(2 * r)],
        [np.cos(r) - 4 * np.sin(2 * r)],
        [6 * np.cos(3 * r)]])

    v = np.matmul(D, np.matmul(Q, Dff))
    return np.linalg.norm(v)

def trefoil_mfd_dist(x, y, integrand=integrand_trefoil):
    xt = np.arcsin(x[2] / 2) / 3
    yt = np.arcsin(y[2] / 2) / 3

    bx = np.asarray([np.abs(xt)])
    by = np.asarray([np.abs(yt)])

    dist = learn_distance(bx, by, np.eye(1), integrand)
    return dist

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

    dist = learn_distance(bx, by, np.eye(2), integrand)
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
