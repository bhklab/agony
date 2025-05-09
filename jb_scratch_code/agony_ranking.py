import networkx as nx
import numpy as np 
import pandas as pd 
import logging
from typing import List, Dict
import argparse 
import yaml
from collections import defaultdict
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
import graph_utils as gu
import tqdm
import os
import graph_agony as ga


def main(config:Dict):


	screen_name, item_type, drug_file, expression_file, response_file, ccl_meta_file =\
		utils.unpack_parameters(config['DATA_PARAMS'])
	
	nbr_min, nbr_max, nbr_step, delta_min, delta_max, delta_count =\
		utils.unpack_parameters(config['EXPERIMENT_PARAMS'])

	nbr_range = np.arange(nbr_min,nbr_max+1, nbr_step)
	
	delta_range = np.linspace(delta_min, delta_max, delta_count)

	with open(drug_file,"r") as istream:
		drug_names = istream.readlines()
		drug_names = [drug_name.rstrip() for drug_name in drug_names]

	seed = 1234
	rng = np.random.default_rng(seed)
	
	expression = pd.read_csv(expression_file)
	response = pd.read_csv(response_file)
	ccl_data = pd.read_csv(ccl_meta_file)
	ccl_to_tissue = {}

	
	for idx,row in ccl_data.iterrows():
		ccl_to_tissue[row['sampleid']]= row['tissueid']
	
	tissues = pd.unique(ccl_data['tissueid'])
	plot_samples = []
	for tissue in tissues:
		samples = ccl_data[ccl_data['tissueid']==tissue]['sampleid'].values
		plot_samples.append(rng.choice(samples,1)[0])
	
	plot_deltas = rng.choice(delta_range,4,replace=False)

	response = response[response['Drug'].isin(drug_names)]
	
	expression = expression[expression['Cell_line'].isin(list(pd.unique(response['Cell_line'])))]
	gs = ["../data/COSMIC.txt"]
	genes = utils.load_gene_sets(gs)

	expression = expression[['Cell_line']+[gene for gene in genes if gene in expression.columns]]

	idx_to_ccl, ccl_to_idx = {},{}

	response.reset_index(inplace=True, drop=True)
	expression.reset_index(inplace=True, drop=True)
	# print(ccl_)
	for idx, row in expression.iterrows():
		idx_to_ccl[idx] = row['Cell_line']
		ccl_to_idx[row['Cell_line']] = idx

	
	X = expression[expression.columns[1:]].values
	X = X/ np.linalg.norm(X,axis=1, keepdims=True)
	# print(X.shape)
	# print(response.columns)
	results = defaultdict(list)
	# plotting_deltas = np.random.choice(delta,)
	for num_nbrs in nbr_range:
		knn_searcher = NearestNeighbors(n_neighbors=num_nbrs)
			
		for delta in delta_range:

			for cl_idx in tqdm.tqdm(np.arange(X.shape[0])):
				if idx_to_ccl[cl_idx] not in plot_samples:
					continue
				X_test = X[cl_idx,:]
				X_train = np.delete(X,cl_idx,axis=0)

				knn_searcher.fit(X_train)
				
				_,nbrs = knn_searcher.kneighbors(X_test.reshape(1,-1))
				nbr_ccls = [idx_to_ccl[j] for j in nbrs[0]]
				
				nbr_graphs,union_graph = gu.build_graphs_from_nbrs(
					response,
					nbr_ccls,
					delta)
				
				# results['ccl'].append(idx_to_ccl[ccl_to_idx])
				# results['delta'].append(delta)
				# results['num_union_edges'].append(len(union_graph.edges))
				
				if idx_to_ccl[cl_idx] in plot_samples and delta in plot_deltas:
					# print("making figs")
					cl_tissue = ccl_to_tissue[idx_to_ccl[cl_idx]]
					cl = idx_to_ccl[cl_idx]
					fig_path = f"../figs/{screen_name}/{cl}/{num_nbrs}/{np.round(delta,2)}/"
					os.makedirs(fig_path,exist_ok=True)
					
					for j in range(len(nbr_graphs)):
						g = nbr_graphs[j]
						pos = graphviz_layout(g,prog='dot')
						nx.draw(g,node_size=5000,pos = pos,with_labels=True)
						# plt.tight_layout()
						plt.savefig(f"{fig_path}nbr_{j}_{nbr_ccls[j]}.png")
						plt.close()
					

					pos = nx.spring_layout(union_graph)
					nx.draw(union_graph,node_size = 5000, pos = pos,with_labels=True)
					# plt.tight_layout()
					plt.savefig(f"{fig_path}union_graph.png")
					plt.close()
					num_cycles = len([x for x in nx.simple_cycles(union_graph)])
					
					union_graph,_,_ = utils.rename_nodes(union_graph,'Drug')
					_,ag = ga.unweighted_agony_lp(union_graph)
					with open(f"{fig_path}meta_data.txt","w") as ostream:
						ostream.write(f"{cl_tissue}\t{num_cycles}")
					
					node_colors = []
					ranks = list(pd.unique([n[1]['rank'] for n in ag.nodes(data=True)]))
					colors = plt.cm.Set2(range(len(ranks)))
					rank_to_color = {ranks[i]:colors[i] for i in range(len(ranks))}
					# colors = [c for c in colors]
					node_labels = {}
					rank_col_map = {}
					for node in ag.nodes(data=True):
						rank = node[1]['rank']
						node_colors.append(rank_to_color[rank])
						node_labels[node[0]]=node[1]['Drug']

					print(node_colors)
					pos = nx.spring_layout(ag)

					f = plt.figure(1)
					ax = f.add_subplot(1,1,1)
					for rank in sorted(rank_to_color.keys()):
						ax.plot([0],[0],color=rank_to_color[rank],label=f"rank {rank}")
					nx.draw(ag, node_size = 5000, pos = pos,
						labels = node_labels, node_color = node_colors)
					leg = plt.legend()
# get the individual lines inside legend and set line width
					for line in leg.get_lines():
						line.set_linewidth(4)
					plt.axis('off')
					f.set_facecolor('w')

					plt.legend()

					f.tight_layout()
					plt.savefig(f"{fig_path}colored_union_graph.png")
					plt.close()
					ranks = nx.get_node_attributes(ag,"rank")
					backward_edges = []
					ag2 = ag.copy()
					for e in ag.edges():
						u,v = e[0], e[1]
						if ranks[u]<= ranks[v]:
							backward_edges.append((u,v))
					ag.remove_edges_from(backward_edges)
					pos =  graphviz_layout(ag,prog='dot')
					nx.draw(ag,node_size = 5000, pos = pos,labels = node_labels, node_color = node_colors)
					plt.savefig(f"{fig_path}colored_union_graph_broken.png")
					plt.close()
					backward_edges = []
					for e in ag2.edges():
						u,v = e[0], e[1]
						if ranks[u]<ranks[v]:
							backward_edges.append((u,v))
					ag.remove_edges_from(backward_edges)
					pos =  graphviz_layout(ag,prog='dot')
					nx.draw(ag,node_size = 5000, pos = pos,labels = node_labels, node_color = node_colors)
					plt.savefig(f"{fig_path}colored_union_graph_broken_fwd.png")
					plt.close()
					
					


	df = pd.DataFrame(results)
	print(df)



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(
		prog = "Graph Agony Drug Ranking",
		description = "Assign a cell line and the kNN")

	parser.add_argument("-config", help = "The configuration file for the experiment.")

	args = parser.parse_args()
	with open(args.config) as config_file:
		config = yaml.safe_load(config_file)

	main(config)