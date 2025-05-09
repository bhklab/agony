import sys
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np 
import pandas as pd 
from typing import List, Dict
import argparse 
import yaml
import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from networkx.drawing.nx_agraph import graphviz_layout



gene_sets = ["kegg.txt", "LINCS.txt"]

genes = utils.load_gene_sets("../data",gene_sets)

expression = pd.read_csv("../data/preprocessed/gCSI/expression.csv")
response = pd.read_csv("../data/preprocessed/gCSI/responses.csv")

print(response['Cell_line'].value_counts())
print(response['Drug'].value_counts())
response = response[response['Drug'].isin(drug_list)]

expression = expression[expression['Cell_line'].isin(list(pd.unique(response['Cell_line'])))]
delta = 0.1

expression = expression[['Cell_line']+[gene for gene in genes if gene in expression.columns]]
idx_to_ccl, ccl_to_idx = {},{}

response.reset_index(inplace=True, drop=True)
expression.reset_index(inplace=True, drop=True)
for idx, row in expression.iterrows():
	idx_to_ccl[idx] = row['Cell_line']
	ccl_to_idx[row['Cell_line']] = idx

X = expression[expression.columns[1:]].values
X = X/ np.linalg.norm(X,axis=1, keepdims=True)
index = np.arange(X.shape[0])
X_train, X_test, train_idx, test_idx = train_test_split(X,index,train_size = 0.99,random_state=0)

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X_train)
_, idx = nbrs.kneighbors(X_test)
print(idx)
_, idx = nbrs.kneighbors(X[test_idx,:])
print(idx)
nbr_ccls = [idx_to_ccl[j] for j in idx[0,:]]

empty_G = nx.DiGraph()
temp = response[response['Cell_line'].isin(nbr_ccls)]
union_graph = nx.DiGraph()

for ccl in nbr_ccls:
	
	this_G = empty_G.copy()
	
	this_temp = response[response['Cell_line']==ccl]
	this_G.add_nodes_from(list(pd.unique(this_temp['Drug'])))
	
	this_drugs = list(pd.unique(this_temp['Drug']))

	for drug1 in this_drugs:
		for drug2 in this_drugs:
			if drug1 == drug2:
				continue
			s1 = this_temp[this_temp['Drug']==drug1]['AAC'].values[0]
			s2 = this_temp[this_temp['Drug']==drug2]['AAC'].values[0]
			
			print(f"Sensitivity for {drug1} is {s1}")
			print(f"Sensitivity for {drug2} is {s2}")
			if s1>s2 and np.abs(s1-s2)>delta:
				this_G.add_edge(drug1,drug2)
				union_graph.add_edge(drug1,drug2)
	# all_graphs.append(this_G)
	pos = graphviz_layout(this_G,prog='dot')
	print(list(nx.simple_cycles(this_G)))
	nx.draw(this_G,pos = pos,with_labels=True)
	plt.title(ccl)
	plt.tight_layout()
	plt.savefig(f"../figs/{ccl}.png")
	plt.show()

# G_ = nx.union_all(all_graphs)
pos = graphviz_layout(union_graph,prog='dot')
print(list(nx.simple_cycles(union_graph)))
nx.draw(union_graph,pos = pos,with_labels=True)
plt.tight_layout()
plt.savefig(f"../figs/union_graph.png")
plt.show()