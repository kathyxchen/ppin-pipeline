"""
This script iterates through a directory of ADAGE model weight matrices and
determines the pathways significant in each model after crosstalk is removed.

Output:
    .txt tab-delimited files specifying the significant pathways
    found in the model.
    Columns:
      model size, node #, side ('pos' or 'neg'), pathway, p-value, q-value

Usage:
	pipeline.py <models-dir> <output-dir>
		<pathway-definitions> <gene-compendium>
		[--cores=<n-cores>] [--replace=<replace>] [--std=<std-signature>]
		[-a | --all-genes]
    pipeline.py -h | --help

Options:
    -h --help                   Show this screen.

    <models-dir>                Path to the directory containing ADAGE models
    <output-dir>                Path to the directory that will store the
                                output files. Will be created if no such path
                                exists.
    <pathway-definitions>       Path to the pathway definitions file.
                                Formatted as tab-delimited columns:
                                pathway, N (num. genes), geneA;geneB;...geneN
    <gene-compendium>           Path to the gene compendium file.

    --replace=<replace>         The resulting file keeps the naming convention
                                of the input file, with the exception of this
                                substring, which will be replaced by 'SigPathways'
                                [default: network]
    --cores=<n-cores>           Number of cores used to run crosstalk removal on
                                models in parallel
                                [default: num. available cores - 1]
    --std=<std-signature>       Number of standard deviations from the mean
                                gene expression value that defines the +/-
                                cutoff for an ADAGE node's high positive weight
                                and high negative weight gene signatures.
                                [default: 2.5]

    -a --all-genes              Apply crosstalk removal to the full gene set
                                in each ADAGE node. By default, the procedure
                                is only applied to genes in the node's gene
                                signatures (genes with a high positive weight
                                or high negative weight).


"""

from docopt import docopt
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
	
	output_filename = weight_file.replace(substring_to_replace, "_SigPathway_")
	output_path = os.path.join(significant_pathways_dir, output_filename)
	significant_pathways_df.to_csv(
		path_or_buf=output_path, sep="\t", index=False)
	return True

if __name__ == "__main__":
	arguments = docopt(
		__doc__, version="eADAGE pathway coverage without crosstalk, 0.0")
	models_directory = arguments["<models-dir>"]
	output_directory = arguments["<output-dir>"]
	pathway_definitions_file = arguments["<pathway-definitions>"]
	gene_compendium = arguments["<gene-compendium>"]
	stddev = float(arguments["--std"])
	substring_to_replace = arguments["--replace"]
	all_genes = arguments["--all-genes"]

	# create the output directory if it does not exist
	try: 
	    os.makedirs(arguments["<output-dir>"])
	except OSError:
	    if not os.path.isdir(arguments["<output-dir>"]):
	        raise

	if arguments["--cores"].isdigit():
		n_cores = int(arguments["--cores"])
	else:
		n_cores = multiprocessing.cpu_count() - 1

	# only need the gene IDs from the compendium file.
	pcl_data = pd.read_csv(gene_compendium, sep="\t", usecols=["Gene_symbol"])
	gene_ids = pcl_data.iloc[:,0]  # format: list(str) of gene IDs
	
	t_o = time()
	np.seterr(all="raise")
	with Parallel(n_jobs=n_cores) as parallel:
		results = parallel(
			delayed(process_model)(weight_file)
			for weight_file in os.listdir(models_directory))
	t_f = time() - t_o
	print("{0} models took {1} seconds to run on {2} cores.".format(
		len(results), t_f, n_cores))
