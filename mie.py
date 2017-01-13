import logging
import pdb
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests

import utils

LOGGER = logging.getLogger("pathway_enrichment")

def pathway_enrichment_without_crosstalk(feature_weight_vector,
                                         alpha, std, n_genes,
                                         genes_in_pathway_definitions,
                                         pathway_definitions_map):
    """Identify positively and negatively enriched pathways in a constructed
    feature.
    
    Parameters
    -----------
    feature_weight_vector : pandas.Series(float), shape = n
        A vector containing gene weights
    alpha : float
        Significance level for pathway enrichment.
    std : float
        Only consider "high weight" genes: genes with weights +/- `std`
        standard deviations from the mean.
    n_genes : int
        The total number of genes in the compendium.
    genes_in_pathway_definitions : set(str)
        The union of all genes in the list of pathway definitions
    pathway_definitions_map : dict(str -> list(str))
        Pathway definitions, pre-crosstalk-removal. A pathway (key) is defined
        by a list of genes (value).
    
    Returns
    -----------
    pandas.DataFrame | None
    """
    mean = feature_weight_vector.mean()
    cutoff = std * feature_weight_vector.std()
    positive_gene_signature = set(
        feature_weight_vector[feature_weight_vector >= mean + cutoff].index)
    negative_gene_signature = set(
        feature_weight_vector[feature_weight_vector <= mean - cutoff].index)
        
    gene_signature = ((positive_gene_signature | negative_gene_signature) & 
                      genes_in_pathway_definitions)

    if not gene_signature:
        return None
    # maximum impact estimation returns positional pathway definitions--
    # each column index (corresponding to a pathway) is mapped to a set of
    # row indices (corresponding to genes)
    new_pathway_definitions = {}
    pathway_index_map = utils.index_element_map(pathway_definitions_map.keys())
    
    row_names = list(genes_in_pathway_definitions)
    row_index_map = utils.index_element_map(row_names)
    membership_matrix = initialize_membership_matrix(
        row_names, pathway_definitions_map)

    remaining_genes = genes_in_pathway_definitions - gene_signature
    crosstalk_removed_pathways = maximum_impact_estimation(membership_matrix)
    new_pathway_definitions = _update_pathway_definitions(
        gene_signature,
        crosstalk_removed_pathways, row_index_map, pathway_index_map,
        new_pathway_definitions)
    for pathway, gene_list in pathway_definitions_map.items():
        if pathway in new_pathway_definitions:
            new_pathway_definitions[pathway] |= (set(gene_list) & remaining_genes)
    
    pathway_positive_series = single_side_pathway_enrichment(
        new_pathway_definitions, positive_gene_signature, n_genes)
    pathway_negative_series = single_side_pathway_enrichment(
        new_pathway_definitions, negative_gene_signature, n_genes)
    pvalue_information = pathway_positive_series.append(pathway_negative_series)

    side_information = pd.Series(["pos"] * len(pathway_positive_series)).append(
        pd.Series(["neg"] * len(pathway_negative_series)))
    side_information.index = pvalue_information.index
    side_information.name = "side"
    return _significant_pathways_dataframe(
        pvalue_information, side_information, alpha)

def _significant_pathways_dataframe(pvalue_information,
                                    side_information,
                                    alpha):
    significant_pathways = pd.concat(
        [pvalue_information, side_information], axis=1)
    below_alpha, fdr_values, _, _ = multipletests(
        significant_pathways["p-value"], alpha=alpha, method="fdr_bh")
    below_alpha = pd.Series(
        below_alpha, index=pvalue_information.index, name="pass")
    fdr_values = pd.Series(
        fdr_values, index=pvalue_information.index, name="padjust")
    significant_pathways = pd.concat(
        [significant_pathways, below_alpha, fdr_values], axis=1)
    significant_pathways = significant_pathways[significant_pathways["pass"]]
    significant_pathways.drop("pass", axis=1, inplace=True)
    significant_pathways.loc[:,"pathway"] = significant_pathways.index
    return significant_pathways

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
      - The keys are ints corresponding to the pathway column indices in the
        membership matrix.
      - The values are int lists corresponding to gene row indices in the
        membership matrix. 
    """
    # Initialize the probability vector as the sum of each column in the
    # membership matrix normalized by the sum of the entire membership matrix.
    # The probability at some index j in the vector represents the likelihood
    # that a pathway (column) j is defined by this set of genes (rows).
    pr_0 = np.longdouble(np.sum(membership_matrix, axis=0) / 
                         np.sum(membership_matrix))
    difference = np.abs(1. - np.sum(pr_0))
    assert difference < 1e-12, "Probabilities sum to {0}.".format(np.sum(pr_0))

    pr_1 = _update_probabilities(pr_0, membership_matrix)
    epsilon = np.linalg.norm(pr_1 - pr_0)/100.
    
    pr_old = pr_1
    check_for_convergence = epsilon
    i = 0  # for logging
    while epsilon > 0. and (check_for_convergence >= epsilon):
        pr_new = _update_probabilities(pr_old, membership_matrix)
        check_for_convergence = np.linalg.norm(pr_new - pr_old)
        pr_old = pr_new
        i += 1
    logging.info("Number of steps taken for EM: {0}".format(i))
    
    pr_final = pr_old  # renaming for readability
    
    new_pathway_definitions = {}
    n, _ = membership_matrix.shape
    for gene_index in range(n):
        gene_membership = membership_matrix[gene_index]
        denominator = np.dot(gene_membership, pr_final)
        # Approximation to prevent divide by zero warning.
        # Since we are only looking for the most probable pathway in which a
        # gene contributes its maximum impact, precision is not as important
        # as maintaining the relative differences between each
        # pathway's probability.
        if denominator < 1e-300:
            denominator = 1e-300
        conditional_pathway_pr = np.multiply(gene_membership, pr_final) / denominator
        pathway_index = np.argmax(conditional_pathway_pr)
        if pathway_index not in new_pathway_definitions:
            new_pathway_definitions[pathway_index] = []
        new_pathway_definitions[pathway_index].append(gene_index)
    return new_pathway_definitions

def _update_probabilities(pr, membership_matrix):
    """Updates the probability vector for each iteration of the
    expectation maximum algorithm in maximum impact estimation.
    
    Parameters
    -----------
    pr : numpy.array(float), shape = [k]
        The current vector of probabilities. An element at index j,
        where j is between 0 and k - 1, corresponds to the probability that,
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
        new_pr = weighted_pathway_col_sums / sum_of_col_sums
    except FloatingPointError:
        # In the event that we encounter underflow or overflow issues, apply
        # this approximation.
        cutoff = 1e-150 / k
        log_cutoff = np.log(cutoff)

        weighted_pathway_col_sums = _replace_zeros(
            weighted_pathway_col_sums, cutoff)
        log_weighted_col_sums = np.log(weighted_pathway_col_sums)
        log_weighted_col_sums -= np.max(log_weighted_col_sums)

        below_cutoff = log_weighted_col_sums < log_cutoff
        geq_cutoff = log_weighted_col_sums >= log_cutoff
        
        logging.info("{1} adjustments made to a vector of length {0}"
                     " containing the raw weight values"
                     " in a call to 'update_probabilities'".format(
                     k, len(log_weighted_col_sums[below_cutoff])))
        
        new_pr = np.zeros(k)
        new_pr[below_cutoff] = cutoff
        col_sums_geq_cutoff = log_weighted_col_sums[geq_cutoff]
        new_pr[geq_cutoff] = np.exp(
            col_sums_geq_cutoff) / np.sum(np.exp(sorted(col_sums_geq_cutoff)))
    
    difference = np.abs(1. - np.sum(new_pr))
    assert difference < 1e-12, "Probabilities sum to {0}.".format(
           np.sum(new_pr))
    return new_pr

def _replace_zeros(arr, default_min_value):
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

def _update_pathway_definitions(gene_signature, index_pathway_definitions,
                                gene_index_map, pathway_index_map,
                                current_pathway_definitions):
    for pathway_index, list_gene_indices in index_pathway_definitions.items():
        pathway = pathway_index_map[pathway_index]
        if pathway not in current_pathway_definitions:
            current_pathway_definitions[pathway] = set()
        genes = [gene_index_map[index] for index in list_gene_indices]
        current_pathway_definitions[pathway] |= (set(genes) & gene_signature)
    return current_pathway_definitions

def single_side_pathway_enrichment(pathway_definitions, gene_signature, n_genes):
    """Identify enriched pathways using the Fisher's exact test for significance
    on a given pathway definition and target gene signature.
    
    Parameters
    -----------
    pathway_definitions : dict(str -> list(str))
        Pathway definitions, *post-crosstalk*-removal. A pathway (key) is
        defined by a list of genes (value). Each gene is only
        associated with one pathway.
    gene_signature : list(str)
        The set of genes we consider to be enriched in a node.
    n_genes : int
        The total number of genes that were considered in the unsupervised
        model.
    
    Returns
    -----------
    pandas.Series, for each pathway, the p-value from applying the Fisher's
                   exact test.
    """
    pvalues_list = []
    for pathway, definition in pathway_definitions.items():
        both_definition_and_signature = len(definition & gene_signature)
        in_definition_not_signature = (len(definition) -
            both_definition_and_signature)
        in_signature_not_definition = (len(gene_signature) -
            both_definition_and_signature)
        neither_definition_nor_signature = (n_genes -
            both_definition_and_signature - in_definition_not_signature -
            in_signature_not_definition)
        contingency_table = np.array(
            [[both_definition_and_signature, in_signature_not_definition],
            [in_definition_not_signature, neither_definition_nor_signature]])
        
        _, pvalue = stats.fisher_exact(
            contingency_table, alternative="greater")
        pvalues_list.append(pvalue)
    pvalues_series = pd.Series(
        pvalues_list, index=pathway_definitions.keys(), name="p-value")
    return pvalues_series

def initialize_membership_matrix(gene_row_names, pathway_definitions_map):
    """Create the binary gene-to-pathway membership matrix that
    will be considered in the maximum impact estimation procedure.
    
    Parameters
    -----------
    gene_row_names : set(str)
        The genes for which we want to assess pathway membership
    pathway_definitions_map : dict(str -> list(str))
        Pathway definitions, pre-crosstalk-removal.
        A pathway (key) is defined by a list of genes (value).
    
    Returns
    -----------
    numpy.array, shape = [n, k], the membership matrix
    """
    membership = []
    for pathway, full_definition in pathway_definitions_map.items():
        pathway_genes = list(full_definition & set(gene_row_names))
        membership.append(np.in1d(gene_row_names, pathway_genes))
    membership = np.array(membership).astype("float").T
    return membership

def pathway_enrichment_with_crosstalk(feature_weight_vector, alpha, n_genes,
                                      genes_in_pathway_definitions,
                                      pathway_definitions_map,
                                      defined_gene_signature):
    """Identify positively and negatively enriched pathways in a constructed
    feature.
    
    Parameters
    -----------
    feature_weight_vector : pandas.Series(float), shape = n
        A vector containing gene weights
    alpha : float
        Significance level for pathway enrichment.
    n_genes : int
        The total number of genes in the compendium.
    genes_in_pathway_definitions : set(str)
        The union of all genes in the list of pathway definitions
    pathway_definitions_map : dict(str -> list(str))
        Pathway definitions, pre-crosstalk-removal. A pathway (key) is defined
        by a list of genes (value).
    defined_gene_signature : functools.partial callable, returns (set(), set())
        Accepts the `feature_weight_vector` as input. Provide a function to
        distinguish +/- signature genes (genes considered to have the greatest
        contribution to a feature's functional signature).
    
    Returns
    -----------
    pandas.DataFrame | None
    """
    positive_gene_signature, negative_gene_signature = defined_gene_signature(
        feature_weight_vector)
    gene_signature = ((positive_gene_signature | negative_gene_signature) & 
                      genes_in_pathway_definitions)

    if not gene_signature:
        return None

    pathway_positive_series = single_side_pathway_enrichment(
        pathway_definitions_map, positive_gene_signature, n_genes)
    pathway_negative_series = single_side_pathway_enrichment(
        pathway_definitions_map, negative_gene_signature, n_genes)
    pvalue_information = pathway_positive_series.append(pathway_negative_series)

    side_information = pd.Series(["pos"] * len(pathway_positive_series)).append(
        pd.Series(["neg"] * len(pathway_negative_series)))
    side_information.index = pvalue_information.index
    side_information.name = "side"
    return _significant_pathways_dataframe(
        pvalue_information, side_information, alpha)