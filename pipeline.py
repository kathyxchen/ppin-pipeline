"""Top-level script"""

import multiprocessing
import os
import pdb
import sys
from time import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

import mie
import utils

### initialize our constants ###  # TODO: replace with argparse
STD_ABOVE = 2.5  # threshold used when identifying our gene signatures
weight_dir = "data/EM300-10DIFF"  # ADAGE weight matrices
significant_pathways_dir = "data/SIGPWS-10DIFF-PIPELINE"  # output directory
gene_compendium = "data/all-pseudomonas-gene-normalized.pcl"  # gene compendium, only used for gene names
pathway_definitions_file = "data/pathway_definitions.txt"  # pathways are defined as a group of genes
substring_to_replace = "_ClusterByweighted_avgweight_"  # when writing to a file

# read in the gene IDs from data file
pcl_data = pd.read_csv(gene_compendium, sep="\t", usecols=["Gene_symbol"])
gene_ids = pcl_data.iloc[:,0]  # format: list(str) of gene IDs

# read in the pathway definitions
pathway_definitions = pd.read_csv(pathway_definitions_file,
	sep="\t", header=None, names=["pw", "size", "genes"], usecols=["pw", "genes"])
pathway_definitions["genes"] = pathway_definitions["genes"].map(lambda x: x.split(";"))
pathway_definitions.set_index("pw", inplace=True)
union_pathway_genes = set()
for gene_list in pathway_definitions["genes"]:
	union_pathway_genes = union_pathway_genes.union(set(gene_list))

pathway_definitions_map = utils.get_pathway_definitions_map(pathway_definitions)

def process_model(weight_file):
	full_filepath = os.path.join(weight_dir, weight_file)
	weight_matrix = utils.load_weight_matrix(full_filepath, gene_ids)
	significant_pathways_df = pd.DataFrame(
		[], columns=["node", "pathway", "p-value", "side", "padjust"])
	for node in weight_matrix:
		node_df = mie.single_node_pathway_enrichment(
			weight_matrix[node], STD_ABOVE, len(gene_ids), union_pathway_genes, pathway_definitions_map)
		node_df.loc[:,"node"] = pd.Series([node] * len(node_df.index), index=node_df.index)
		significant_pathways_df = pd.concat([significant_pathways_df, node_df], axis=0)
	significant_pathways_df.reset_index(drop=True, inplace=True)
	
	print(significant_pathways_df.shape)
	output_filename = weight_file.replace(substring_to_replace, "_SigPathway_")
	output_path = os.path.join(significant_pathways_dir, output_filename)
	significant_pathways_df.to_csv(
		path_or_buf=output_path, sep="\t", index=False)
	return True

if __name__ == "__main__":
	t_o = time()
	np.seterr(all="raise")
	n_cores = multiprocessing.cpu_count()
	with Parallel(n_jobs=n_cores-1) as parallel:
		results = parallel(delayed(process_model)(weight_file) for weight_file in os.listdir(weight_dir))
	t_f = time() - t_o
	print("{0} models took {1} seconds to run on {2} cores.".format(len(results), t_f, n_cores-1))

