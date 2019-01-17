# ----------------------------------------------------------------------------------------------------
#
# RANDOM TREE GENERATION AND VISUALIZATION
#
# ----------------------------------------------------------------------------------------------------

polblogs1 = scipy.io.loadmat('./data/realnet/karate_data_1_edges.mat')
intEdges = polblogs1['edges']  ##

Edges = [[str(i) for i in Edge] for Edge in intEdges]

max_size = 0
for Edge in intEdges.tolist():
    for i in Edge:
        if i > max_size:
            max_size = i

# print(Edges)
Labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#
#################################################################################################################

# ZACHARY'S KARATE CLUB DATASET

'''
Labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Edges = [[str(i) for i in Edge] for Edge in intEdges]
'''


# import matlab.engine

# matlab = matlab.engine.start_matlab()
#
# # Node types: pref, unif, bal
# # Label types: unif, hier
# def generate_tree(nodes, type, label):
#     return matlab.gen_rand_tree(nodes, type, 2, label, nargout=4)
#
# Tree = generate_tree(20, 'pref', 'hier')
# Edges = [[str(int(node)) for node in edge] for edge in Tree[0]]
# Labels = [int(label[0]) for label in Tree[3]]

# print(Labels)
# print(Edges)
