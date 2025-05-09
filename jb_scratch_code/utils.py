import numpy as np
import sys
from typing import List, Dict, Union
import networkx as nx
import pandas as pd
from collections import defaultdict
from itertools import combinations, chain




def rename_nodes(
    G:nx.Graph,
    new_field_name:str = 'Gene'
    ):
    gene_to_idx  = {} 
    idx_to_gene = {}
    for idx, gene in enumerate(G.nodes):
        gene_to_idx[gene] = idx
        idx_to_gene[idx] = gene
    G = nx.relabel_nodes(G,gene_to_idx)
    nx.set_node_attributes(G,idx_to_gene, new_field_name)
    return G, gene_to_idx, idx_to_gene


def unpack_parameters(
    D:Dict
    ):
    if len(D.values())>1:
        return tuple(D.values())
    else:
        return tuple(D.values())[0]


# def load_gene_sets(
#     gene_set_dir:str,
#     gene_set_fnames:List[str]
#     ) -> List[str]:
    
#     union_gene_set = []

#     for gene_set_file in gene_set_fnames:
#         with open(f"{gene_set_dir}/{gene_set_file}", "r") as istream:
#             genes = istream.readlines()
#         genes = [gene.rstrip() for gene in genes]
#         genes = [gene for gene in genes if gene not in union_gene_set]
#         union_gene_set.extend(genes)
#     return union_gene_set
def load_gene_sets(
    gene_set_fnames:List[str]
    ) -> List[str]:
    
    union_gene_set = []

    for gene_set_file in gene_set_fnames:
        with open(gene_set_file, "r") as istream:
            genes = istream.readlines()
        genes = [gene.rstrip() for gene in genes]
        genes = [gene for gene in genes if gene not in union_gene_set]
        union_gene_set.extend(genes)
    return union_gene_set