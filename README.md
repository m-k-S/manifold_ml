Manifold Metric Learning
=====

## Dependencies

Our code was written and tested using Python 3.6.7; other versions may work but no guarantees are made. Libraries used are:
- numpy 1.15.4
- scipy 1.2.0
- scikit-learn 0.20.2
- networkx 2.2 [only for MDS]

The code should run out of the box, assuming dependencies are installed. Please open an issue if you encounter any errors in setup.

## Usage Guide

#### Metric Learning Performance Tests

To run clustering tests on metric learning, use the following syntax:

```
python3 metric_learning.py --dataset DATASET --lmbd LAMBDA --reg REG --clus
```

Similarly, for classification, use the following:

```
python3 metric_learning.py --dataset DATASET --K k --lmbd LAMBDA --reg REG --clf
```

k is used in classification tests to specify the k in k-nearest neighbor. Reg should be a float between 0.0 and 1.0 and specifies the regularization term in the loss function. Lambda should be a float, and specifies how much scaling is penalized during optimization. Dataset names available out of the box are: football, polbooks, karate, adjnoun, helicoid and 20newsgroup.

#### Metric Learning

To perform manifold metric learning on your dataset of choice, import one of the loss functions (`mmc_loss_generic` or `lmnn_loss_generic`) and run the following optimization (for example):

```
Q0 = numpy.diag([1 for _ in range(dim)])
res = scipy.optimize.minimize(lmnn_loss_generic, Q0, args=(None, K, reg, lmbd, fxn_mfd, fxn_mfd_dist, fxn_integrand, data, labels), method='Powell')
```

`Q0` should be the initial linear transformation; in our work we have generally set it to the identity matrix. `fxn_mfd` is a function that takes points in the base Euclidean space and maps them to the manifold (as described in the paper). `fxn_mfd_dist` is the distance function on that manifold; if no explicit distance function exists, an approximation function can be created using `learn_manifold_distance.py` (and the appropriate `fxn_integrand` must be set). `data` should be an array of points in the base space, and `labels` should be a corresponding array of point labels.

#### Manifold Distance Approximation

To approximate the distance function on a manifold, one must compute the arc length integrand as a function of paths (as described in the paper). Plugging this integrand into the function `learn_distance` in `learn_manifold_distance.py` will yield an approximation of the true manifold distance (see paper for empirical rate of convergence)

### MDS

Minimizing the MDS loss function in `generalized_mds.py` requires the functions `fxn_mfd`, `fxn_mfd_dist`, and (optionally) `fxn_integrand`, as described above. It also requires a nxn distance matrix D (where n is the number of points to be embedded, and D<sub>ij</sub> is the distance between the ith and jth points), an initialized matrix of embedded points (which may be generated randomly), and the desired embedding dimension.
