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
		[-a | --all-genes] [-v | --verbose]
    pipeline.py -h | --help

Options:
    -h --help                   Show this screen.
    -v --verbose                Output logging information.

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
from __future__ import *
import logging
import multiprocessing
import os
import pdb  # TODO: remove when script is finalized.
import sys
from time import time

from docopt import docopt
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

import mie
import utils

OUTPUT_FILE_NAMING = "SigPathway"

class ProcessModel:

	def __init__(gene_ids, pathway_definitions_map, union_pathway_genes,
				 std, use_all_genes):
		self.gene_ids = gene_ids

		self.pathway_definitions_map = pathway_definitions_map
		self.union_pathway_genes = union_pathway_genes

		self.std_signature = std
		self.use_all_genes = use_all_genes

	def process(self, models_directory, current_model):
		full_filepath = os.path.join(models_directory, current_model)
		weight_matrix = utils.load_weight_matrix(full_filepath, self.gene_ids)
		significant_pathways_df = pd.DataFrame(
			[], columns=["node", "pathway", "p-value", "side", "padjust"])
		n_genes = len(self.gene_ids)
		for node in weight_matrix:
			node_df = mie.single_node_pathway_enrichment(
				weight_matrix[node], self.std_signature, n_genes,
				self.union_pathway_genes, self.pathway_definitions_map,
				self.use_all_genes)
			node_df.loc[:,"node"] = pd.Series([node] * len(node_df.index), index=node_df.index)
			significant_pathways_df = pd.concat([significant_pathways_df, node_df], axis=0)
		significant_pathways_df.reset_index(drop=True, inplace=True)
		return (weight_file, significant_pathways_df)

if __name__ == "__main__":
	arguments = docopt(
		__doc__, version="eADAGE pathway coverage without crosstalk, 0.0")
	models_directory = arguments["<models-dir>"]
	output_directory = arguments["<output-dir>"]
	pathway_definitions_file = arguments["<pathway-definitions>"]
	gene_compendium = arguments["<gene-compendium>"]
	std = float(arguments["--std"])
	substring_to_replace = arguments["--replace"]
	all_genes = arguments["--all-genes"]
	verbose = arguments["--verbose"]

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

	logger = logging.getLogger("pathway_coverage_without_crosstalk")
	if verbose:
		logger.setLevel(logging.INFO)

	gene_ids = utils.load_gene_identifiers(gene_compendium)
	all_defined_genes, pathway_definitions = utils.load_pathway_definitions(
		pathway_definitions_file)

	process_model = ProcessModel(
		gene_ids, pathway_definitions, all_defined_genes, std, all_genes)
	
	t_o = time()
	np.seterr(all="raise")
	with Parallel(n_jobs=n_cores) as parallel:
		results = parallel(
			delayed(process_model.process)(models_directory, model)
			for model in os.listdir(models_directory))
	t_f = time() - t_o
	print("{0} models took {1} seconds to run on {2} cores.".format(
		len(results), t_f, n_cores))

	for model, significant_pathways_df in results:
		output_filename = model.replace(substring_to_replace, OUTPUT_FILE_NAMING)
		output_path = os.path.join(output_directory, output_filename)
		significant_pathways_df.to_csv(
			path_or_buf=output_path, sep="\t", index=False)
