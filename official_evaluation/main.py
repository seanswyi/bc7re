#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:32:08 2021

@author: tonifuc3m

Modified by seanswyi.
"""
import argparse
import os
import warnings

import pandas as pd

from compute_metrics import compute_metrics
from utils import format_relations, get_chemical_gene_combinations, load_entities_dict, preprocess_data


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def parse_arguments():
    '''
    DESCRIPTION: Parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.add_argument('-g', '--gs_path', required=False, dest='gs_path', default='/hdd1/seokwon/data/BC7DP/drugprot-gs-training-development/development', help='path to GS relations file (TSV)')
    parser.add_argument('-e', '--entity_path', required=False, dest='entity_path', default='../gs-data/gs_entities.tsv', help='path to GS entities file (TSV)')
    parser.add_argument('-p', '--pred_path', required=False, dest='pred_path', default='../toy-data/pred_relations.tsv', help='path to predictions file (TSV)')
    parser.add_argument('--pmids', required=False, dest='pmids', default='../gs-data/pmids.txt', help='path to list of valid pubmed IDs. One PMID per line')

    return parser.parse_args()


def main(args):
    '''
    Load GS and Predictions; format them; compute precision, recall and
    F1-score and print them.

    Parameters
    ----------
    gs_path : str
        Path to GS Relations TSV file.
    pred_path : str
        Path to Predictions Relations TSV file.
    ent_path : str
        Path to GS Entities TSV file
    pmids : str
        Path to file with valid pubmed IDs

    Returns
    -------
    None.
    '''
    relation_names = ['INDIRECT-DOWNREGULATOR',
                      'INDIRECT-UPREGULATOR',
                      'DIRECT-REGULATOR',
                      'ACTIVATOR',
                      'INHIBITOR',
                      'AGONIST',
                      'AGONIST-ACTIVATOR',
                      'AGONIST-INHIBITOR',
                      'ANTAGONIST',
                      'PRODUCT-OF',
                      'SUBSTRATE',
                      'SUBSTRATE_PRODUCT-OF',
                      'PART-OF']

    relname2tag = {name: idx + 1 for idx, name in enumerate(relation_names)}
    num_relations = len(relname2tag)

    # Load basic data.
    print("Loading GS files...")
    pmid2chemicals_and_genes, _, chemicals = load_entities_dict(args.entity_path)
    pmid2combinations, num_combinations = get_chemical_gene_combinations(pmid2chemicals_and_genes)
    pmids = set(map(lambda x: str(x.strip()), open(args.pmids)))

    # Load GS data.
    gs = pd.read_csv(args.gs_path, sep='\t', header=None, dtype=str, skip_blank_lines=True, names=['pmid', 'rel_type', 'arg1', 'arg2'], encoding='utf-8')

    # Load predictions.
    print("Loading prediction files...")
    preds = pd.read_csv(args.pred_path, sep='\t', header=None, dtype=str, skip_blank_lines=True, names=['pmid', 'rel_type', 'arg1', 'arg2'], encoding='utf-8')

    # Format data
    print("Checking GS files...")
    gs_valid, gs_rel_list = preprocess_data(df=gs, chemicals=chemicals, rel_types=relation_names, is_gs=True)

    print("Checking Predictions files...")
    preds_valid, preds_rel_list = preprocess_data(df=preds, chemicals=chemicals, rel_types=relation_names, gs_files=pmids)

    y_true, y_pred = format_relations(gs_valid=gs_valid,
                                      preds_valid=preds_valid,
                                      pmid2combinations=pmid2combinations,
                                      num_combinations=num_combinations,
                                      num_relations=num_relations,
                                      relname2tag=relname2tag)

    # Compute metrics
    print("Computing DrugProt (BioCreative VII) metrics ...\n(p = Precision, r=Recall, f1 = F1 score)")
    compute_metrics(y_true=y_true, y_pred=y_pred, relname2tag=relname2tag, gs_rel_list=gs_rel_list, preds_rel_list=preds_rel_list)


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(args.gs_path):
        raise Exception(f'Gold Standard path {args.gs_path} does not exist')

    if not os.path.exists(args.pred_path):
        raise Exception(f'Predictions path {args.pred_path} does not exist')

    if not os.path.exists(args.entity_path):
        raise Exception(f'Gold Standard entities path {args.entity_path} does not exist')

    if not os.path.exists(args.pmids):
        raise Exception(f'PMIDs file list path {args.pmids} does not exist')

    main(args)
