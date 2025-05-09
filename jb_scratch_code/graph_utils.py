import pandas as pd
import networkx as nx 
import numpy as np
from typing import List, Tuple


def build_graphs_from_nbrs(
	affinity_scores:pd.DataFrame,
	nbrs:List[str],
	delta: float,
	voter_col:str = 'Cell_line',
	item_col:str = 'Drug',
	affinity_col:str = 'AAC'
	)-> Tuple[List[nx.DiGraph],nx.DiGraph]:
	
	graph_list = []
	empty_G = nx.DiGraph()
	# print(affinity_scores.head())
	affinities = affinity_scores[affinity_scores[voter_col].isin(nbrs)]
	union_graph = nx.DiGraph()
	# this_G = empty_G.copy()
	
	for voter in nbrs:
		temp = affinities[affinities[voter_col]== voter]
		_items = list(pd.unique(temp[item_col]))
		item_graph = nx.DiGraph()
		
		
		item_graph.add_nodes_from(_items)
	

		for item1 in _items:
			for item2 in _items:
				if item1 == item2:
					continue
				
				s1 = temp[temp[item_col]==item1][affinity_col].values[0]
				s2 = temp[temp[item_col]==item2][affinity_col].values[0]
				
				if s1>s2 and np.abs(s1-s2)>delta:
					item_graph.add_edge(item1,item2)
					union_graph.add_edge(item1,item2)
	
		graph_list.append(item_graph)
	return graph_list, union_graph


# def construct_union_graph(
# 	graph_list:List[nx.Digraph]
# 	)->nx.Digraph:
	
# 	