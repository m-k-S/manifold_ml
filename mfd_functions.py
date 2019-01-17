import numpy as np

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

def hyp_mfd_dist(x,y, integrand):  # takes data in mfd space NOT BASE SPACE
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


# MAP TO MFD
def map_dataset_to_mfd(B, Q, mfd_generic):
    FQB = []
    for x in B:
        # x = x[1:]  <<<< DO NOT UNCOMMENT!
        Qx = np.matmul(Q, x)
        mQx = mfd_generic(Qx)
        FQB.append(mQx)

    return FQB
