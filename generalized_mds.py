# ----------------------------------------------------------------------------------------------------
#
# GENERALIZED SURFACE MULTIDIMENSIONAL SCALING
#
# ----------------------------------------------------------------------------------------------------

import numpy as np

# mfd_generic is a function that maps points in the base space B to the specified manifold
# mfd_dist_generic is the distance function on the manifold
#   if the distance function is not explicitly known, it can be approximated using the functions in learn_distance.py
def mds_loss(B, npts, dim, Dist, mfd_generic, mfd_dist_generic, integrand):
    # Dist is a symmetric npts x npts distance matrix
    # We are optimizing over location of the points in the base space B

    B = B.reshape(npts, dim)  # Datapoints in base space B,  B is the variable of optimization
    I = np.diag([1 for _ in range(dim)])

    FB = map_dataset_to_mfd(B, I, mfd_generic)

    loss = 0
    for i in range(npts):
        for j in range(i-1):  # Traverse the upper triangular matrix
            loss += (mfd_dist_generic(FB[i], FB[j], integrand) - Dist[i][j]) ** 2

    print(B)
    return loss

# Random initialization of points in base space using standard multivariate Gaussians
def mds_initialization(npts, dim):
    pts = []
    mean = np.zeros(dim)
    cov = np.diag([1 for _ in range(dim)])
    for i in range(npts):
        pts.append(np.random.multivariate_normal(mean, cov))

    return np.asarray(pts)

if __name__ == "__main__":
    import argparse
    from manifold_functions import hyp_mfd, hyp_mfd_dist
    from scipy.optimize import minimize
    import networkx as nx
    import scipy.io


    parser = argparse.ArgumentParser()
    parser.add_argument('--gml', help='path to GML file containing network edges')
    parser.add_argument('--mat', help="path to MAT file containing network edges; variable name must be 'edges'")
    parser.add_argument('--dim', help="desired embedding dimension")
    args = parser.parse_args()

    # Modify these to embed into a specific manifold
    mfd_generic == None
    mfd_dist_generic == None
    mfd_integrand == None

    if mfd_generic == None:
        print("No generalized surface specified; will embed into the hyperboloid by default")
        mfd = hyp_mfd
        mfd_dist = hyp_mfd_dist
        mfd_integrand = None

    if args.dim:
        dim = int(args.dim)
    else:
        print("No embedding dimension specified.")
        exit()

    if args.gml:
        H = nx.read_gml(args.gml)
        G = nx.convert_node_labels_to_integers(H)
    elif args.mat:
        edges = scipy.io.loadmat(args.mat)['edges']
        G = nx.Graph()
        G.add_edges_from(edges)
    else:
        print("No file containing edges specified, exiting.")
        exit()

    max_size = G.order()
    distance_matrix = [[0 for _ in range(max_size)] for _ in range(max_size)]
    for i in range(max_size):
        for j in range(max_size):
            try:
                distance_matrix[i][j] = len(nx.shortest_path(G, i, j)) - 1
            except nx.exception.NetworkXNoPath:
                distance_matrix[i][j] = 50

    B0 = mds_initialization(max_size, dim)
    mds_Powell_minimize = minimize(mds_loss, B0, args=(max_size, dim, discrete_metric, mfd, mfd_dist, mfd_integrand), method='Powell', options={'disp': True})
    Bnew = mds_Powell_minimize.x.reshape(max_size, dim)
    print(Bnew)
