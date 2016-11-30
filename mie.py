import logging
import pdb
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests

import utils

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

def update_probabilities(pr, membership_matrix):
    """Updates the probability vector for each iteration of the
    expectation maximum algorithm in maximum impact estimation.
    
    Parameters
    -----------
    pr : numpy.array(float), shape = [k]
        The current vector of probabilities. An element at index j,
        where j is between 0 and k-1, corresponds to the probability that,
        given a gene g_i, g_i has the greatest impact in pathway j.
    membership_matrix : numpy.array(float), shape = [n, k]
        The observed gene-to-pathway membership matrix, where n is the number
        of genes and k is the number of pathways we are interested in.
        
    Returns
    -----------
    numpy.array(float), shape = [k], a vector of updated probabilities
    """
    n, k = membership_matrix.shape
    pathway_col_sums = np.sum(membership_matrix, axis=0)
    
    weighted_pathway_col_sums = np.multiply(pathway_col_sums, pr)
    sum_of_col_sums = np.sum(weighted_pathway_col_sums)
    try:
        new_pr = weighted_pathway_col_sums/sum_of_col_sums
    except FloatingPointError:
        cutoff = 1e-150/k
        log_cutoff = np.log(cutoff)

        weighted_pathway_col_sums = utils.replace_zeros(weighted_pathway_col_sums, cutoff)
        log_wpcs = np.log(weighted_pathway_col_sums)
        max_log_wpcs = np.max(log_wpcs)
        log_wpcs = log_wpcs - max_log_wpcs
        logging.info("{1} adjustments made to a vector of length {0} containing the raw weight values"
                     " in a call to 'update_probabilities'".format(
                len(log_wpcs),len(log_wpcs[log_wpcs < log_cutoff])))
        
        new_pr = np.zeros(len(log_wpcs))
        new_pr[log_wpcs < log_cutoff] = cutoff
        geq_cutoff = log_wpcs >= log_cutoff
        new_pr[geq_cutoff] = np.exp(log_wpcs[geq_cutoff])/np.sum(np.exp(sorted(log_wpcs[geq_cutoff])))
    # make sure the probabilities sum to 1
    difference = np.abs(1. - np.sum(new_pr))
    assert difference < 1e-12, "Probabilities sum to {0}.".format(np.sum(new_pr))
    return new_pr

def maximum_impact_estimation(membership_matrix):
    """Determines the underlying pathway impact matrix:
    each gene is mapped to the pathway in which it has the most impact.
    
    Parameters
    -----------
    membership_matrix : numpy.array(float), shape = [n, k]
        The observed gene-to-pathway membership matrix, where n is the number
        of genes and k is the number of pathways we are interested in.
    
    Returns
    -----------
    dict(int -> list(int)), a dictionary mapping a pathway to a list of genes.
    These are the new pathway definitions after the maximum impact estimation
    procedure has been applied to remove crosstalk. 
      - The keys are ints corresponding to the pathway column indices in the membership matrix.
      - The values are int lists corresponding to gene row indices in the membership matrix. 
    """
    pr_0 = np.longdouble(np.sum(membership_matrix, axis=0)/np.sum(membership_matrix))
    difference = np.abs(1. - np.sum(pr_0))
    assert difference < 1e-12, "Probabilities sum to {0}.".format(np.sum(pr_0))

    pr_1 = update_probabilities(pr_0, membership_matrix)
    epsilon = np.linalg.norm(pr_1 - pr_0)/100.
    
    pr_old = pr_1
    check_for_convergence = epsilon
    i = 0  # for logging
    #pdb.set_trace()
    while epsilon > 0. and (check_for_convergence >= epsilon):
        pr_new = update_probabilities(pr_old, membership_matrix)
        check_for_convergence = np.linalg.norm(pr_new - pr_old)
        pr_old = pr_new
        i += 1
    logging.info("Number of steps taken for EM: {0}".format(i))
    # after convergence criterion is met, we have our final probabilities.
    pr_final = pr_old  # renaming for readability
    
    new_pathway_definitions = {}
    n, _ = membership_matrix.shape
    for gene_index in range(n):
        gene_membership = membership_matrix[gene_index]
        denominator = np.dot(gene_membership, pr_final)
        if denominator < 1e-200:
            denominator = 1e-200
        conditional_pathway_pr = gene_membership/denominator
        pathway_index = np.argmax(conditional_pathway_pr)
        if pathway_index not in new_pathway_definitions:
            new_pathway_definitions[pathway_index] = []
        new_pathway_definitions[pathway_index].append(gene_index)
    return new_pathway_definitions

def get_new_pathways(pathway_signature_genes, pathway_remaining_genes,
                     signature_index_map, remaining_index_map, pathway_index_map):
    new_pathway_definitions = {}
    new_pathway_definitions = update_pathway_definitions(
        pathway_signature_genes, signature_index_map, pathway_index_map, new_pathway_definitions)
    new_pathway_definitions = update_pathway_definitions(
        pathway_remaining_genes, remaining_index_map, pathway_index_map, new_pathway_definitions)
    for pathway, definition in new_pathway_definitions.items():
        new_pathway_definitions[pathway] = set(definition)
    return new_pathway_definitions

def update_pathway_definitions(index_pathway_definitions, gene_index_map, pathway_index_map, current_pathway_definitions):
    for pathway_index, list_gene_indices in index_pathway_definitions.items():
        pathway = pathway_index_map[pathway_index]
        if pathway not in current_pathway_definitions:
            current_pathway_definitions[pathway] = []
        genes = [gene_index_map[index] for index in list_gene_indices]
        current_pathway_definitions[pathway] += genes
    return current_pathway_definitions

def single_side_pathway_enrichment(pathway_definitions, gene_signature, n_genes):
    """Identify enriched pathways using the Fisher's exact test for significance
    on a given pathway definition and target gene signature.
    
    Parameters
    -----------
    pathway_definitions : dict(str -> list(str))
        Pathway definitions, *post-crosstalk*-removal. A pathway (key) is defined by a list of genes (value).
        Each gene is only associated with one pathway.
    gene_signature : list(str)
        The set of genes we consider to be enriched in a node.
    n_genes : int
        The total number of genes that were considered in the unsupervised model.
    
    Returns
    -----------
    pandas.Series, for each pathway, the p-value from applying the Fisher's exact test.
    """
    pvalues_list = []
    for pathway, definition in pathway_definitions.items():
        both_definition_and_signature = len(definition & gene_signature)
        in_definition_not_signature = len(definition) - both_definition_and_signature
        in_signature_not_definition = len(gene_signature) - both_definition_and_signature
        neither_definition_nor_signature = (n_genes - both_definition_and_signature -
                                            in_definition_not_signature - in_signature_not_definition)
        contingency_table = np.array([[both_definition_and_signature, in_signature_not_definition],
                                      [in_definition_not_signature, neither_definition_nor_signature]])
        
        _, pvalue = stats.fisher_exact(contingency_table, alternative="greater")
        pvalues_list.append(pvalue)
    pvalues_series = pd.Series(pvalues_list, index=pathway_definitions.keys(), name="p-value")
    return pvalues_series

def single_node_pathway_enrichment(
    node_data, std_above, n_genes, genes_in_pathway_definitions, pathway_definitions_map):
    """Identify positively and negatively enriched pathways in a node.
    
    Parameters
    -----------
    node_data : pandas.Series
        The weight of each gene in this node.
    std_above : int
        Only consider "high weight" genes: genes with weights +/- <std_above> standard deviations from the mean.
    n_genes : int
        The total number of genes in the compendium.
    genes_in_pathway_definitions : set(str)
        The union of all genes in the list of pathway definitions
    pathway_definitions_map : dict(str -> list(str))
        Pathway definitions, pre-crosstalk-removal. A pathway (key) is defined by a list of genes (value).
    
    Returns
    -----------
    TODO
    """
    mean = node_data.mean()
    cutoff = std_above * node_data.std()
    positive_gene_signature = set(node_data[node_data >= mean + cutoff].index)
    negative_gene_signature = set(node_data[node_data <= mean - cutoff].index)
    
    gene_signature = (positive_gene_signature | negative_gene_signature) & genes_in_pathway_definitions
    if len(gene_signature) > 0:
        remaining_genes = genes_in_pathway_definitions - gene_signature

        # crosstalk_removal
        signature_row_names = list(gene_signature)
        remaining_row_names = list(remaining_genes)
        sig_matrix, rem_matrix = initialize_membership_matrices(
            signature_row_names, remaining_row_names, pathway_definitions_map)
        #pdb.set_trace()
        pathway_signature_genes = maximum_impact_estimation(sig_matrix)
        #pdb.set_trace()
        pathway_remaining_genes = maximum_impact_estimation(rem_matrix)

        # 'maximum_impact_estimation' outputs dictionaries with the index information
        # for the pathways and genes.
        signature_index_map = utils.index_element_map(signature_row_names)
        remaining_index_map = utils.index_element_map(remaining_row_names)

        assert [tup[0] for tup in pathway_definitions_map.items()] == list(pathway_definitions_map.keys()), \
               "dictionary items() and keys() are not in the same order"
        pathway_index_map = utils.index_element_map(pathway_definitions_map.keys())

        new_pathway_definitions = get_new_pathways(pathway_signature_genes, pathway_remaining_genes,
                                                   signature_index_map, remaining_index_map,
                                                   pathway_index_map)

        positive_series = single_side_pathway_enrichment(new_pathway_definitions, positive_gene_signature, n_genes)
        negative_series = single_side_pathway_enrichment(new_pathway_definitions, negative_gene_signature, n_genes)
        pvalue_information = positive_series.append(negative_series)

        side_information = pd.Series(["pos"] * len(positive_series)).append(pd.Series(["neg"] * len(negative_series)))
        side_information.index = pvalue_information.index
        side_information.name = "side"

        significant_pathways_df = pd.concat([pvalue_information, side_information], axis=1)
        below_alpha, fdr_values, _, _ = multipletests(significant_pathways_df["p-value"], alpha=0.05, method="fdr_bh")
        below_alpha = pd.Series(below_alpha, index=pvalue_information.index, name="pass")
        fdr_values = pd.Series(fdr_values, index=pvalue_information.index, name="padjust")
        significant_pathways_df = pd.concat([significant_pathways_df, below_alpha, fdr_values], axis=1)
        significant_pathways_df = significant_pathways_df[significant_pathways_df["pass"]]
        significant_pathways_df.drop("pass", axis=1, inplace=True)
        significant_pathways_df.loc[:,"pathway"] = significant_pathways_df.index
        return significant_pathways_df
    return None

def initialize_membership_matrices(signature_row_names, remaining_row_names, pathway_definitions_map):
    """Create the binary gene-to-pathway membership matrices that will be considered in 
    the maximum impact estimation procedure.
    
    Parameters
    -----------
    signature_row_names : set(str)
        The genes in a node gene signature
    remaining_row_names : set(str)
        The genes outside of a node gene signature
    pathway_definitions_map : dict(str -> list(str))
        Pathway definitions, pre-crosstalk-removal. A pathway (key) is defined by a list of genes (value).
    
    Returns
    -----------
    tuple(np.array, np.array), shape = [n, k], the membership matrices
    for pathway genes (1) in a node gene signature and (2) outside of a node gene signature. 
    """
    signature_membership = []
    remaining_membership = []
    
    for pathway, full_definition in pathway_definitions_map.items():
        pathway_gene_signature = list(full_definition & set(signature_row_names))
        signature_membership.append(np.in1d(signature_row_names, pathway_gene_signature))
        
        remaining_genes = list(full_definition & set(remaining_row_names))
        assert set(remaining_genes) == full_definition - set(pathway_gene_signature)
        remaining_membership.append(np.in1d(remaining_row_names, remaining_genes))
    
    signature_membership = np.array(signature_membership).astype("float").T
    remaining_membership = np.array(remaining_membership).astype("float").T
    return (signature_membership, remaining_membership)