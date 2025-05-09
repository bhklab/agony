import cvxpy
import numpy as np
import abc
import logging
import networkx as nx
import cvxopt as cx
import networkx as nx
import numpy as np



def unweighted_agony_lp(
    graph:nx.DiGraph
    )->nx.DiGraph:
    
    r"""
    
    Compute the graph agony for an unweighted directed graph by solving the linear program (LP)
    defined in the original agony paper [1]. 

    For a directed graph G = (V,E) we use the shorthand:
        - :math:
            n=|V|
        - :math:
            m = |E|


    [1] Gupte, Mangesh, et al. "Finding hierarchy in directed online social networks." 
        Proceedings of the 20th international conference on World wide web. 2011.


    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    
    # We use this vector in the objective function to sum over the edges
    c = [1. for i in range(m)] + [0. for i in range(n)]

    # matrix encoding linear relationships
    G = [[0. for i in range(2 * (m + n))] for j in range(n + m)]
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
    
  

    node_to_rank = {}
    for i in graph.nodes:
        node_to_rank[i]=int(np.round(solution['x'][-n:][i]))
    
    nx.set_node_attributes(graph,node_to_rank,"rank")
    solution = int(round(solution['primal objective']))
    # print(solution)
    # print(1-solution/m)
    return solution, graph

    
class RankerBase: def __init__( self, delta:float, )

class AgonyRanker:
	def __init__(
		self,
		G:nx.DiGraph,
		weighted:bool=False,
		accelerated:bool=False,
		weight:str=None
		)->None:

	def fit(self
		)->None:
		pass

	def compute_agony(
		self,
		)-> None:
		pass
		# return

	def get_ranks(
		self)->None:
		pass


	def break_cycles(
		self,
		)->None:

		"""
		Heuristic Cycle Breaking from 
		"""
