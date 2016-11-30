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