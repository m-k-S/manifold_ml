# ----------------------------------------------------------------------------------------------------
#
# POINCARE EMBEDDING (AND CONVERSION TO HYPERBOLOID)
#
# ----------------------------------------------------------------------------------------------------

# from gensim.models.poincare import PoincareModel, PoincareRelations
# from gensim.test.utils import datapath
# from gensim.viz.poincare import poincare_2d_visualization
from os import getcwd


def plot_embedding(model, labels, title, filename):
    plot = poincare_2d_visualization(model, labels, title)
    py.plot(plot, filename=filename)

'''
# REAL DATA

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
# DEAR LORD! TOO MUCH MAGIC!

# model = PoincareModel(Edges, negative=10, size=DIMENSION)
# model.train(epochs=50)
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

# B = model.kv.vectors
# B = b2h_Matrix(B)
# print(B)

# scipy.io.savemat('polblogs_hyp.mat', mdict = {'arr': B})

# B = [np.asarray(i[1:]) for i in B]

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

S = []
# for edge in S_Labels:
    # x = b2h_Vector(model.kv.__getitem__(edge[0]))
    # y = b2h_Vector(model.kv.__getitem__(edge[1]))
    # S.append([x, y])
    # S.append([B[edge[0]], B[edge[1]]])

D = []
# for edge in D_Labels:
    # x = b2h_Vector(model.kv.__getitem__(edge[0]))
    # y = b2h_Vector(model.kv.__getitem__(edge[1]))
    # D.append([x, y])
    # D.append([B[edge[0]], B[edge[1]]])


lip = lambda x, y : -x[0] * y[0] + sum([x[i] * y[i] for i in range(1, len(x))])
dhyp = lambda x, y : np.arccosh(-lip(x, y))

# print ("Distance between pair 1 on Poincare ball: " + str(model.kv.distance(S_Labels[0][0], S_Labels[0][1])))
# print ("Distance between pair 1 on hyperboloid: " + str(dhyp(S[0][0], S[0][1])))
