"""
Utility functions
"""
import numpy as np
import pandas as pd


def load_weight_matrix(path_to_weight_file, gene_ids):
	"""Reads in the ADAGE model weight matrix.
	
	Parameters
	-----------
	path_to_weight_file : string
	gene_ids : pandas.Series
		
	Returns
	-----------
	pandas.DataFrame weight matrix, indexed on the gene IDs
	"""
	weight_matrix = pd.read_csv(
		path_to_weight_file, sep="\t", header=None, skiprows=2, nrows=len(gene_ids))
	weight_matrix["gene_ids"] = pd.Series(gene_ids, index=weight_matrix.index)
	weight_matrix.set_index("gene_ids", inplace=True)
	return weight_matrix

def load_gene_identifiers(path_to_gene_compendium):
	"""
	Parameters
	-----------
	path_to_gene_compendium : string
		The path to the gene compendium file on which ADAGE was trained
		
	Returns
	-----------
	list(str) a list of gene identifiers
	"""
	pcl_data = pd.read_csv(path_to_gene_compendium, sep="\t", usecols=["Gene_symbol"])
	gene_ids = pcl_data.iloc[:,0]
	return gene_ids

def load_pathway_definitions(path_to_pathway_definitions):
	"""
	Parameters
	-----------
	path_to_pathway_definitions : string
		The path to the pathway definitions file
		
	Returns
	-----------
	tuple(set(str), dict(str -> set(str)))
	(1) The set of genes present in at least 1 pathway definition
	(2) A dictionary of pathway definitions, where a pathway (key) is mapped
	    to a set of genes (value) 
	"""
	pathway_definitions = pd.read_csv(path_to_pathway_definitions,
		sep="\t", header=None, names=["pw", "size", "genes"], usecols=["pw", "genes"])
	pathway_definitions["genes"] = pathway_definitions["genes"].map(lambda x: x.split(";"))
	pathway_definitions.set_index("pw", inplace=True)
	union_pathway_genes = set()
	for gene_list in pathway_definitions["genes"]:
		union_pathway_genes = union_pathway_genes.union(set(gene_list))
	pathway_definitions_map = get_pathway_definitions_map(pathway_definitions)
	return union_pathway_genes, pathway_definitions_map

def get_pathway_definitions_map(pathway_definitions_df):
	"""
	Parameters
	-----------
	pathway_definitions_df : pandas.DataFrame
		columns = [pathway, [geneA, geneB, ..., geneN]]
		index column = pathway
		
	Returns
	-----------
	dict(str -> set(str)), a pathway (key) is mapped to a set of genes (value).
	"""
	pathway_definitions_map = {}
	for index, row in pathway_definitions_df.iterrows():
		pathway_definitions_map[index] = set(row["genes"])
	return pathway_definitions_map

def label_trim(full_pw_label):
	""" Based on the naming conventions for GO and KEGG pathways,
	shorten the pathway labels if possible.

	#TODO: provide some examples of the naming convention.
	"""
	if full_pw_label[0:2] == "GO":
		trim_idx = full_pw_label.index(":")
		if trim_idx != -1:
			return full_pw_label[trim_idx + 1:] + " - GO"
		else:
			return full_pw_label
	else:
		to_remove = "- Pseudomonas aeruginosa PAO1"
		pw_trim = full_pw_label.split(" ", 1)
		if len(pw_trim) > 1:
			pw_trim = pw_trim[1]
		if to_remove in pw_trim:
			return pw_trim[0:pw_trim.index(to_remove)] + "PA01"
		elif ":" in full_pw_label:
			return pw_trim
		else:
			return full_pw_label.strip()

def read_significant_pathways_file(path_to_file):
	node_pathway_df = pd.read_table(path_to_file,
									sep="\t",
									header=0,
									usecols=["node", "side", "pathway"])
	node_pathway_df["pathway"] = node_pathway_df["pathway"].apply(
		lambda x: label_trim(x))
	node_pathway_df = node_pathway_df.sort_values(by=["node", "side"])
	return node_pathway_df

def index_element_map(arr):
	"""Map the indices of the array to the respective elements.
	
	Parameters
	-----------
	arr : list(a)
		The array to process, of generic type a
		
	Returns
	-----------
	dict(int -> a), a dictionary corresponding the index to the element
	"""
	index_to_element = {}
	for index, element in enumerate(arr):
		index_to_element[index] = element
	return index_to_element
