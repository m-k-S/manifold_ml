import numpy as np
import scipy.io
from config import username, api_key
import plotly

plotly.tools.set_credentials_file(username=username, api_key=api_key)

# ----------------------------------------------------------------------------------------------------
#
# RANDOM TREE GENERATION AND VISUALIZATION
#
# ----------------------------------------------------------------------------------------------------

import collections
import plotly.plotly as py
import plotly.graph_objs as go
import matlab.engine

matlab = matlab.engine.start_matlab()

# Node types: pref, unif, bal
# Label types: unif, hier
def generate_tree(nodes, type, label):
    return matlab.gen_rand_tree(nodes, type, 2, label, nargout=4)

Tree = generate_tree(20, 'pref', 'hier')
Edges = [[str(int(node)) for node in edge] for edge in Tree[0]]
Labels = [int(label[0]) for label in Tree[3]]

# print(Labels)
# print(Edges)

# ----------------------------------------------------------------------------------------------------
#
# POINCARE EMBEDDING (AND CONVERSION TO HYPERBOLOID)
#
# ----------------------------------------------------------------------------------------------------

from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath
from gensim.viz.poincare import poincare_2d_visualization
from os import getcwd

DIMENSION = 2

def plot_embedding(model, labels, title, filename):
    plot = poincare_2d_visualization(model, labels, title)
    py.plot(plot, filename=filename)

# file_path = datapath('')
# model = PoincareModel(PoincareRelations(file_path), negative=2, size=DIMENSION)
# model.train(epochs=50)

'''
REAL DATA

# polblogs1 = scipy.io.loadmat('./data/realnet/polblogs_data_1.mat')
# print(polblogs1.keys())
'''

'''
edges = []
randnet = open(os.getcwd() + "/data/randnet/psmodel_deg4_gma2.25_10_net.txt", "r")
for line in randnet:
    line = line.split("\n")[0]
    line = line.split(" ")
    edges.append([line[0].rstrip(), line[1].rstrip()])

# labels = [str(i) for i in range(1, 501)]
model = PoincareModel(edges, negative=2, size=DIMENSION)
model.train(epochs=200)
plot_embedding(model, edges, "Test Graph", "test14")

'''

model = PoincareModel(Edges, negative=10, size=DIMENSION)
model.train(epochs=600)
# plot_embedding(model, Edges, "Test Graph", "Test 20")

# Parameters:
#   n x D matrix B (n points, D dimension) [ie transposed]
# Returns:
#   n x D+1 matrix
def b2h_Matrix(B):
    B = np.asarray(B)
    # x0 is a 1xn dim vector
    x0 = (2. / (1 - np.sum(np.power(B, 2), axis=1))) - 1
    x = np.multiply(B.T, x0 + 1)

    return np.vstack((x0, x)).T

def b2h_Vector(v):
    x0 = (2. / (1 - np.sum(np.power(v, 2)))) - 1
    x = np.multiply(v, x0 + 1)
    return np.hstack((x0, x))

B = model.kv.vectors
B = b2h_Matrix(B)

S_Labels = []
D_Labels = []
for i in range(1, len(Labels) + 1):
    for j in range(1, len(Labels) + 1):
        if i != j:
            if Labels[i-1] == Labels[j-1]:
                if not ([str(i), str(j)] in S_Labels) and not ([str(j), str(i)] in S_Labels):
                    S_Labels.append([str(i), str(j)])
            else:
                if not ([str(i), str(j)] in D_Labels) and not ([str(j), str(i)] in D_Labels):
                    D_Labels.append([str(i), str(j)])

# print(S)

S = []
for edge in S_Labels:
    x = b2h_Vector(model.kv.__getitem__(edge[0]))
    y = b2h_Vector(model.kv.__getitem__(edge[1]))
    S.append([x, y])

print ("DISTANCE BTWN PAIR 1: " + str(model.kv.distance(S_Labels[0][0], S_Labels[0][1])))

lip = lambda x, y : -x[0] * y[0] + sum([x[i] * y[i] for i in range(1, len(x))])
dhyp = lambda x, y : np.arccosh(-lip(x, y))
print ("sanity check - dist on hyperboloid btwn pair 1 is: " + str(dhyp(S[0][0], S[0][1])))

D = []
for edge in D_Labels:
    x = b2h_Vector(model.kv.__getitem__(edge[0]))
    y = b2h_Vector(model.kv.__getitem__(edge[1]))
    D.append([x, y])

# ----------------------------------------------------------------------------------------------------
#
# LEARNING M
#
# ----------------------------------------------------------------------------------------------------

from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import quad

minkowski_diagonal = [1 for _ in range(DIMENSION+1)]
minkowski_diagonal[0] = -1
minkowski_metric_tensor = np.diag(minkowski_diagonal)

# Require that (x-y)^T Q^T G Q (x - y) > 0 otherwise we don't have a true sense of distance (ie want PSD Q)

def Negatives(x, n):
    samples = []
    for edge in D:
        w = x == edge[0]
        v = x == edge[1]
        if w.all() or v.all():
            samples.append(edge)
        if len(samples) == n:
            break

    return samples

NEGATIVES = 5

def L(Q):
    G = minkowski_metric_tensor
    ip = lambda x, y : np.matmul(x.T, np.matmul(Q.T, np.matmul(G, np.matmul(Q, y))))
    Q = Q.reshape(DIMENSION+1, DIMENSION+1)

    total = 0
    for edge in S:
        x = edge[0]
        y = edge[1]

        num = 1. / (-ip(x, y) + np.sqrt(ip(x, y)**2 - 1))

        denom = 0
        samples = Negatives(x, NEGATIVES)
        for sample in samples:
            denom += 1. / (-ip(sample[0], sample[1]) + np.sqrt(ip(sample[0], sample[1])**2 - 1))

        total += num / denom

    return total

def gradL(Q):
    G = minkowski_metric_tensor
    Q = Q.reshape(DIMENSION+1, DIMENSION+1)

    ip = lambda x, y : np.matmul(x.T, np.matmul(Q.T, np.matmul(G, np.matmul(Q, y))))
    dz = lambda x, y : np.matmul(np.matmul(G, np.matmul(Q, x)), y.T) + np.matmul(np.matmul(G, np.matmul(Q, y)), x.T)
    df = lambda x, y : (-1 / (-ip(x, y) + np.sqrt(ip(x, y) ** 2 - 1)) ** 2) * (-1 + (ip(x, y) / np.sqrt(ip(x, y) ** 2 - 1))) * (dz(x, y))

    total = 0
    for edge in S:
        x = edge[0]
        y = edge[1]
        num = df(x, y)

        denom = 0
        samples = Negatives(x, NEGATIVES)
        for sample in samples:
            denom += df(sample[0], sample[1])

        total += num / denom

    return total

def learn_distance(x, y, Q, samples=100, segments=100):
    def integrand(t, x, y, Q):
        Pth = (1 - t) * x + (y * t)
        Dff = y-x
        ip = np.matmul(Dff.T, np.matmul(Q.T, np.matmul(Q, Dff)))
        nm = np.matmul(Pth.T, Dff)
        dn = 1 + np.matmul(Pth.T, Pth)

        return np.sqrt( ip + (nm**2 / dn) )
#### PERHAPS SWITCH TO EUCLIDEAN DISTANCE FOR TESTING PURPOSES?? IGNORE Q
#### in distance calculation

    x = x[1:]
    y = y[1:]
    path_segments = []
    for i in range(segments):
        path_segments.append((i / (segments-1) * (y - x)) + x)
### DO RANOMISH INITIALIZATION RATHER THAN LINSPACE INITIALIZATION?  SO THAT WE CAN RECOVER BACK
### STRAIGHT LINE BACK (IE DONT START WITH STRAIGHT LINE)
### ONE WAY IS TO ADD RANDOM JITTER TO EACH SEGMENT INITIALLY

    convergence = True

    while convergence is True:
        convergence = False

        # make sure this loop goes  length(path_segments) -2 stimes  idx=1 to idx=length(path_segments)-2 (inclusive)
        # this is what is happening
        for idx, segment in enumerate(path_segments[1:-1]):
            i = idx + 1
            this_point = segment
            prev_point = path_segments[i-1]
            next_point = path_segments[i+1]

            sample_radius = max(np.linalg.norm(this_point - prev_point), np.linalg.norm(this_point - next_point))
            #sample_radius = max(np.linalg.norm(segment - path_segments[idx-1]), np.linalg.norm(segment - path_segments[idx+1])) / 2

            # Generate random vectors and normalize them
            s = np.random.uniform(-sample_radius,sample_radius,size=(samples, DIMENSION))


            # D1 = quad(integrand, 0, 1, args=(prev_point, this_point, Q))[0]
            # D2 = quad(integrand, 0, 1, args=(this_point, next_point, Q))[0]
            this_point_x0 = np.sqrt(1 + np.dot(this_point.T, this_point))
            next_point_x0 = np.sqrt(1 + np.dot(next_point.T, next_point))
            prev_point_x0 = np.sqrt(1 + np.dot(prev_point.T, prev_point))

            # this_point_hyp = np.concatenate((this_point_x0, this_point))
            # next_point_hyp = np.concatenate((next_point_x0, next_point))
            # prev_point_hyp = np.concatenate((prev_point_x0, prev_point))

            this_point_hyp = [this_point_x0, this_point[0], this_point[1]]
            next_point_hyp = [next_point_x0, next_point[0], next_point[1]]
            prev_point_hyp = [prev_point_x0, prev_point[0], prev_point[1]]

            # this_point_hyp.insert(0, this_point_x0)
            # next_point_hyp.insert(0, next_point_x0)
            # prev_point_hyp.insert(0, prev_point_x0)

            D1 = dhyp(prev_point_hyp, this_point_hyp)
            D2 = dhyp(this_point_hyp, next_point_hyp)

            min_dist = D1+D2
            best_sample = this_point  # maybe nan?
            for sample in s:
                # Add sample to current point
                sample = sample + this_point
                # Check distance on manifold from current point to sample point


                # D1 = quad(integrand, 0, 1, args=(prev_point, sample, Q))[0]
                # D2 = quad(integrand, 0, 1, args=(sample, next_point, Q))[0]
                sample_x0 = np.sqrt(1 + np.dot(sample.T, sample))
                # sample_hyp = np.concatenate((sample_x0, sample))
                sample_hyp = [sample_x0, sample[0], sample[1]]

                # sample_hyp.insert(0, sample_x0)

                D1 = dhyp(prev_point_hyp, sample_hyp)
                D2 = dhyp(sample_hyp, next_point_hyp)

                Distance = D1 + D2
                if Distance < min_dist:
                    min_dist = Distance
                    best_sample = sample

                    total_distance = 0
                    for ti in range(len(path_segments[:-1])):
                        tdist = quad(integrand, 0, 1, args=(path_segments[ti], path_segments[ti+1], Q))[0]
                        total_distance += tdist
                    print('>>>>' + str(total_distance))


            if np.linalg.norm(this_point - best_sample) < 10e-6:
                path_segments[i] = best_sample
            else:
                convergence = True
                path_segments[i] = best_sample

    # while loop ends

    print(path_segments)

    total_distance = 0
    for idx in range(len(path_segments[:-1])):
        Distance = quad(integrand, 0, 1, args=(path_segments[idx], path_segments[idx+1], Q))[0]
        total_distance += Distance

    return total_distance

def compute_distance(x, y, Q):
    x = x[1:]
    y = y[1:]
    def integrand(t, x, y, Q):
        Pth = (1 - t) * x + y * t
        Dff = y-x
        ip = lambda x, y : np.matmul(Dff.T, np.matmul(Q.T, np.matmul(Q, Dff)))
        nm = lambda x, y : np.matmul(Pth.T, Dff)
        dn = lambda x, y : 1 + np.matmul(Pth.T, Pth)

        # print (ip(x, y) + (nm(x,y) ** 2 / dn(x, y)))
        return np.sqrt( ip(x, y) + (nm(x,y)**2 / dn(x, y)) )

    Distance = quad(integrand, 0, 1, args=(x, y, Q))
    return Distance[0]

def grad_distance(x, y, Q):
    x = x[1:]
    y = y[1:]
    def integrand(t, x, y, Q):
        A = (1 - t) * x + y * t
        ip = lambda x, y : np.matmul((y-x).T, np.matmul(Q.T, np.matmul(Q, y-x)))
        nm = lambda x, y : np.matmul(A.T, x-y)
        dn = lambda x, y : 1 + np.matmul(A.T, A)

        dz = 1. / ( 2 * (np.sqrt( ip(x, y) + (nm(x,y) ** 2 / dn(x, y)) )) )
        df = 2 * np.matmul(np.matmul(Q, x-y), (x-y).T)

        return dz * df

    grad_Distance = quad(integrand, 0, 1, args=(x, y, Q))
    return grad_Distance[0] # returns a n x n matrix

def Loss2(Q):
    Q = Q.reshape(DIMENSION, DIMENSION)
    print(Q)
    total = 0
    for edge in S:
        x = edge[0]
        y = edge[1]
        dQ = learn_distance(x, y, Q)

        total += dQ

    for edge in D:
        x = edge[0]
        y = edge[1]
        dQ = learn_distance(x, y, Q)

        total -= dQ

    return total

def gradL2(Q):
    Q = Q.reshape(DIMENSION, DIMENSION)
    total = 0
    for edge in S:
        x = edge[0]
        y = edge[1]
        dQ = grad_distance(x, y, Q)

        total += dQ

    for edge in D:
        x = edge[0]
        y = edge[1]
        dQ = grad_distance(x, y, Q)

        total -= dQ

    return total

Q0 = np.diag([1 for _ in range(DIMENSION)])
# Q0 = minkowski_metric_tensor

new_x = np.array([1, 0, 0])
new_y = np.array([np.sqrt(5), 2, 0])

# print (learn_distance(new_x, new_y, Q0, 1000, 7))
# print (dhyp(new_x, new_y))

print (learn_distance(S[0][0], S[0][1], Q0, 1000, 7))
print (compute_distance(S[0][0], S[0][1], Q0))

# res_NelderMead = minimize(Loss2, Q0, method='nelder-mead', options={'xtol': 1e-3, 'disp': True})
# print(res_NelderMead)

# res_BFGS = minimize(Loss2, Q0, method='BFGS', jac=gradL2, options={'disp': True})
# print(res_BFGS)

# ----------------------------------------------------------------------------------------------------
#
# PLOT DATA
#
# ----------------------------------------------------------------------------------------------------

Initial_Data = B


# ----------------------------------------------------------------------------------------------------
#
# COMPUTE DISCRETE METRIC
#
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
#
# RECONSTRUCT TREE
#
# ----------------------------------------------------------------------------------------------------
