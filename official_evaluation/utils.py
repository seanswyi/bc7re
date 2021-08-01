#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:32:25 2021

@author: antonio

Modified by seanswyi.
"""
import itertools
import warnings

from tqdm import tqdm


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def save_entity_id(chemicals, genes, entity_type, entity_id):
    if entity_type=='CHEMICAL':
        chemicals.append(entity_id)
    elif entity_type=='GENE':
        genes.append(entity_id)
    else:
        warnings.warn("Wrong entity type.")

    return chemicals, genes


def update_dict(chemicals, genes, pmid, input_dict):
    temp_dict = {}

    temp_dict['chemicals'] = chemicals
    temp_dict['genes'] = genes

    input_dict[pmid] = temp_dict

    return input_dict


def load_entities_dict(path):
    """
    Load entities TSV

    Returns
    -------
    genes : set
        Set of GENE
    chemicals : set
        Set of CHEMICAL
    _dict_ : dict
        Dictionary with annotated entities. Keys: PMID, Values: another
        dictionary with keys 'chemicals' and 'genes' and value the
        annotation mark
    """
    pmid2chemicals_and_genes = {}
    pmid_entity_id2entity_type = {}
    previous_pmid = ''
    chemicals = []
    genes = []

    with open(file=path) as f:
        lines = f.readlines()

        for line in lines:
            info = line.split('\t')

            if (len(info) != 6) and (line != '\n'):
                raise Exception(f"Line {line} in file {path} is wrongly formatted.")

            pmid = info[0]
            entity_id = info[1]
            entity_type = info[2]

            if pmid != previous_pmid:
                pmid2chemicals_and_genes = update_dict(chemicals=chemicals, genes=genes, pmid=previous_pmid, input_dict=pmid2chemicals_and_genes)
                chemicals = []
                genes = []

            chemicals, genes = save_entity_id(chemicals=chemicals, genes=genes, entity_type=entity_type, entity_id=entity_id)
            previous_pmid = pmid
            pmid_entity_id2entity_type[pmid + '-' + entity_id] = entity_type

        pmid2chemicals_and_genes = update_dict(chemicals=chemicals, genes=genes, pmid=previous_pmid, input_dict=pmid2chemicals_and_genes)
        del pmid2chemicals_and_genes[''] # Key for first `previous_pmid`.

        genes = set(key for key, value in pmid_entity_id2entity_type.items() if value == 'GENE')
        chemicals = set(key for key, value in pmid_entity_id2entity_type.items() if value == 'CHEMICAL')

        return pmid2chemicals_and_genes, genes, chemicals


def get_chemical_gene_combinations(input_dict):
    """
    Parameters
    ----------
    _dict_ : dictionary
        Dictionary with annotated entities. Keys: PMID, Values: another
        dictionary with keys 'chemicals' and 'genes' and value the
        annotation mark

    Returns
    -------
    combinations : dictionary
        PMIDs as keys, all possible CHEMICAL-GENE combinations are values.
    NCOMB : int
        DESCRIPTION.
    """
    combinations = {}
    num_combinations = 0

    for pmid, entities in input_dict.items():
        chem = entities['chemicals']
        genes = entities['genes']

        combinations[pmid] = list(itertools.product(chem, genes))
        num_combinations += len(combinations[pmid])

    return combinations, num_combinations


def preprocess_data(df, chemicals, rel_types, is_gs=False, gs_files=None):
    """
    Preprocess annotations dataframe

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe with annotations (GS or predicted).
    chemicals : list
        List of GS CHEMICAL entities.
    rel_types : list
        List of valid relation types.
    is_gs : bool, optional
        Whether we are formatting the GS annotations. The default is False.
    gs_files : set, optional
        Set of PMIDs. The default is set().

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    df : pandas DataFrame
        Clean annotations DataFrame.
    """
    # if df.shape[0] == 0:
    #     raise Exception('There are not parsed annotations')

    if df.shape[1] != 4:
        raise Exception('Wrong column number in the annotations file')

    # Drop duplicates
    df = df.drop_duplicates(subset=df.columns).copy()

    # Remove predictions for RELATION FILES not valid
    unique_input_relations = set(df.rel_type.tolist())
    unique_relation_names = set(rel_types)

    if len(unique_input_relations.intersection(unique_relation_names)) > len(unique_relation_names):
        warnings.warn("Non-valid relations types exist. Skipping them.")

    df = df.loc[df['rel_type'].isin(rel_types), :].copy()

    # Remove predictions for PMIDs not valid
    if not is_gs:
        df = df.loc[df['pmid'].isin(gs_files), :].copy()

    # Check every relation has one CHEMICAL and one GENE
    df['pmid-arg1'] = df['pmid'] + '-' + df['arg1'].apply(lambda x: x.split(':')[-1])
    df['pmid-arg2'] = df['pmid'] + '-' + df['arg2'].apply(lambda x: x.split(':')[-1])
    df['is_arg1_chemical'] = df['pmid-arg1'].apply(lambda x: x in chemicals)
    df['is_arg2_chemical'] = df['pmid-arg2'].apply(lambda x: x in chemicals)

    skip_chem = []
    skip_gene = []

    if df.shape[0] != 0:
        if any(df.apply(lambda x: x['is_arg1_chemical'] + x['is_arg2_chemical'], axis=1) > 1):
            skip_chem = df.loc[df.apply(lambda x: x['is_arg1_chemical'] + x['is_arg2_chemical'], axis=1) > 1].index.tolist()
            warnings.warn(f"The following lines have more than one CHEMICAL entity: {df.loc[skip_chem]}. Skipping them")

        if any(df.apply(lambda x: x['is_arg1_chemical'] + x['is_arg2_chemical'], axis=1) == 0):
            skip_gene = df.loc[df.apply(lambda x: x['is_arg1_chemical'] + x['is_arg2_chemical'], axis=1) == 0].index.tolist()
            warnings.warn(f"The following lines have less than one CHEMICAL entity: {df.loc[skip_gene]}. Skipping them")

        skip = skip_chem + skip_gene

        if len(skip) > 1:
            df.drop(skip, inplace=True)

        df['chemical'] = df.apply(lambda x: x['arg1'].split(':')[-1] if x['is_arg1_chemical'] else x['arg2'].split(':')[-1], axis=1)
        df['gene'] = df.apply(lambda x: x['arg2'].split(':')[-1] if x['is_arg1_chemical'] else x['arg1'].split(':')[-1], axis=1)
    elif df.shape[0] == 0:
        df['chemical'] = []
        df['gene'] = []

    # Keep only relevant columns
    df = df[['pmid', 'rel_type', 'chemical', 'gene']].drop_duplicates(subset=['pmid', 'rel_type', 'chemical', 'gene']).copy()

    return df, set(df.rel_type.tolist())


def format_relations(gs_valid, preds_valid, pmid2combinations, num_combinations, num_relations, relname2tag):
    """
    Format relation information for sklearn

    Parameters
    ----------
    gs_valid : pandas DataFrame
        DESCRIPTION.
    pred_valid : pandas DataFrame
        DESCRIPTION.
    combinations : TYPE
        Possible entity combinations (CHEMICAL-GENE).
    NCOMB : int
        Number of entity combinations (CHEMICAL-GENE).
    NREL : int
        Number of valid relations.
    reltype2tag : dict
        Mapping from relation type string to integer tag.

    Returns
    -------
    y_true : nested lists
        List of GS relations.
    y_pred : nested lists
        List of Predictions relations.
    """
    y_true = [[0] * num_combinations for _ in range(num_relations)]
    y_pred = [[0] * num_combinations for _ in range(num_relations)]
    current_idx = 0

    pbar = tqdm(iterable=sorted(pmid2combinations.items()), desc="Formatting data", total=len(pmid2combinations))
    for pmid, combinations in pbar:
        if combinations == []:
            continue

        # Subset GS and predictions
        gs_relevant_pmid = gs_valid.loc[gs_valid['pmid'] == pmid, :]
        pred_relevant_pmid = preds_valid.loc[preds_valid['pmid'] == pmid, :]

        # Iterate over all combinations
        for idx, combination in enumerate(combinations):
            chemical = combination[0]
            gene = combination[1]

            gs_rel = gs_relevant_pmid.loc[(gs_relevant_pmid['chemical'] == chemical) & (gs_relevant_pmid['gene'] == gene), 'rel_type'].values
            if len(gs_rel) > 0:
                tag = relname2tag[gs_rel[0]]
                y_true[tag - 1][current_idx + idx] = 1

            pred_rel = pred_relevant_pmid.loc[(pred_relevant_pmid['chemical'] == chemical) & (pred_relevant_pmid['gene'] == gene), 'rel_type'].values
            if len(pred_rel) > 0:
                tag = relname2tag[pred_rel[0]]
                y_pred[tag - 1][current_idx + idx] = 1

        current_idx += idx

    return y_true, y_pred
