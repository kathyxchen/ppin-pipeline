import numpy as np
import pandas as pd

def get_pathway_definitions_map(pathway_definitions_df):
    pathway_definitions_map = {}
    for index, row in pathway_definitions_df.iterrows():
        pathway_definitions_map[index] = set(row["genes"])
    return pathway_definitions_map

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

def label_trim(full_pw_label):
    """ Quick code to make the pathway labels shorter. Not especially
        elegant.
    """
    if full_pw_label[0:2] == "GO":
        trim_idx = full_pw_label.index(":")
        if trim_idx != INV:
            return full_pw_label[trim_idx+1:] + " - GO"
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

def load_weight_matrix(weight_file, gene_ids):
    """Reads in the ADAGE model weight matrix.
    
    Parameters
    -----------
    weight_file : string
        The path to the file
    gene_ids : pandas.Series
        
    Returns
    -----------
    pandas.DataFrame weight matrix, indexed on the gene IDs
    """
    weight_matrix = pd.read_csv(weight_file, sep="\t", header=None, skiprows=2, nrows=len(gene_ids))
    weight_matrix["gene_ids"] = pd.Series(gene_ids, index=weight_matrix.index)
    weight_matrix.set_index("gene_ids", inplace=True)
    return weight_matrix

def read_significant_pathways_file(path_to_file):
    node_pathway_df = pd.read_table(path_to_file,
                                    sep="\t",
                                    header=0,
                                    usecols=["node", "side", "pathway"])
    node_pathway_df["pathway"] = node_pathway_df["pathway"].apply(
        lambda x: label_trim(x))
    node_pathway_df = node_pathway_df.sort_values(by=["node", "side"])
    return node_pathway_df

def replace_zeros(arr, default_min_value):
	"""Substitute 0s in the list with a near-zero value.

	Parameters
	-----------
	arr : numpy.ndarray
	default_min_value : float
		If the smallest non-zero element in `arr` is greater than the default,
		use the default instead.

	Returns
	-----------
	numpy.ndarray
	"""
	min_nonzero_value = min(default_min_value, np.min(arr[arr > 0]))
	closest_to_zero = np.nextafter(min_nonzero_value, min_nonzero_value - 1)
	arr[arr == 0] = closest_to_zero
	return arr