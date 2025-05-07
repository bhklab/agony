import os

import cvxopt as cx
import networkx as nx
import numpy as np
import random

random.seed(42)


# is very slow (may not be slow in practice, anyway only using this to test the c++ code)
def slow_agony(graph):
    # use this as example and to understand the code
    # https://cvxopt.org/userguide/coneprog.html#cvxopt.solvers.lp

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    # the variables in our case are p(u, v), r(u), r(v) (as in the paper)
    # the first m variables are for p(u,v)
    # the next n variables are for r(u)

    # we now want to minimize the sum of agony of edges only
    c = [1. for i in range(m)] + [0. for i in range(n)]

    # we have 2*(m + n) equations with n+m variables giving a (n+m)x(2*(m+n)) matrix
    G = [[0. for i in range(2 * (m + n))] for j in range(n + m)]

    # the first m equations
    edge_list = list(graph.edges)
    for i, (u, v) in enumerate(edge_list):
        # we want the conditions
        # r(v) - r(u) - p(u, v) <= -1
        G[m + v][i] = 1.
        G[m + u][i] = -1.
        G[i][i] = -1.
    for i in range(m, 2 * m):
        # these are for the conditions -1 * p(u, v) <= 0
        G[i - m][i] = -1.
    # we add 2*n extra conditions for bounding r(u)
    for i in range(2 * m, 2 * m + n):
        # conditions for r(u) <= n
        G[i - m][i] = 1.
    for i in range(2 * m + n, 2 * m + 2 * n):
        # conditions for -1 * r(u) <= 0
        G[i - m - n][i] = -1.
    c = cx.matrix(c)
    G = cx.matrix(G)
    h = cx.matrix([-1. for i in range(m)] + [0. for i in range(m)] +
                  [0. + n for i in range(n)] + [0. for i in range(n)])
    solution = cx.solvers.lp(c, G, h)
    return int(round(solution['primal objective']))


os.system("g++ -std=c++11 agony.cc agony_test.cc -o agony_test")

TESTS = 100
failed = 0

for t in range(5, TESTS):
    print("Running test #{:d}".format(t))
    n = t
    print("   constructing graph...")
    graph = nx.fast_gnp_random_graph(
        n, 0.8, random.randint(10, 1000), directed=True)
    wId = open("test.in", "w")
    wId.write("{:d} {:d}\n".format(graph.number_of_nodes(),
                                   graph.number_of_edges()))
    for a, b in graph.edges:
        wId.write("{:d} {:d}\n".format(a, b))
    wId.close()
    print("   done\n   running slow LP algorithm...")
    ans_slow = slow_agony(graph)
    print("   done")
    ans_custom = 0
    print("   running faster algorithm...")
    os.system("./agony_test < test.in")
    print("   done")
    with open("agony_temp.txt") as fId:
        ans_custom = int(fId.readline().strip())
    if not ans_slow == ans_custom:
        failed += 1
    else:
        print("   test passed")
    if failed > 0:
        print("   {:d} tests failed".format(failed))
if failed > 0:
        print("{:d} tests failed".format(failed))

os.system("rm agony_test agony_temp.txt")
