import os

import networkx as nx
import numpy as np

os.system("g++ -std=c++11 cycle_dfs_test.cc -o cycle_dfs_test")

TESTS = 20

for t in range(3, TESTS):
    n = t * t
    for k in range(10):
        print("Running test #{:d}".format(t * 10 + k))
        g = nx.fast_gnp_random_graph(n, 0.3, directed=True)
        wId = open("test.in", "w")
        wId.write("{:d} {:d}\n".format(g.number_of_nodes(),
                                       g.number_of_edges()))
        for a, b in g.edges:
            wId.write("{:d} {:d}\n".format(a, b))
        wId.close()
        os.system("cycle_dfs_test < test.in > test.out")
        mat = np.loadtxt("test.out")
        row_sum = mat.sum(axis=0)
        col_sum = mat.sum(axis=1)
        if not np.array_equal(row_sum, col_sum):
            print("now you fucked up")
            print(mat)
