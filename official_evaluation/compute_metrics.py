#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:32:08 2021

@author: tonifuc3m
"""
import warnings

from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix, precision_score, recall_score


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def compute_metrics(y_true, y_pred, relname2tag, gs_rel_list, preds_rel_list):
    '''
    Compute precision, recall and F1-score and print them.

    Parameters
    ----------
    y_true : list
        List of GS Relations.
    y_pred : list
        List of Predictions Relations.
    reltype2tag : dict
        Mapping from relation type string to integer tag
    gs_rel_list : list
        List of relation types in GS
    pred_rel_list : list
        List of relation types in Predictions

    Returns
    -------
    None.
    '''
    relations_not_in_gs = set(relname2tag.keys()) - set(gs_rel_list)
    relations_not_in_pred = set(relname2tag.keys()) - set(preds_rel_list)

    print("By relation type")
    for relation_name, relation_id in relname2tag.items():
        if relation_name in relations_not_in_gs:
            continue

        y_true_this = y_true[relation_id - 1]
        y_pred_this = y_pred[relation_id - 1]

        tn, fp, fn, tp = confusion_matrix(y_true_this, y_pred_this).ravel()

        precision = precision_score(y_true_this, y_pred_this, zero_division=0)
        recall = recall_score(y_true_this, y_pred_this, zero_division=0)
        f1 = f1_score(y_true_this, y_pred_this, zero_division=0)

        # assert tp / (tp + fp + 1e-10) == precision, f"tp / (tp + fp) = {tp / (tp + fp)}\tprecision = {precision}"
        # assert tp / (tp + fn + 1e-10) == recall, f"tp / (tp + fp) = {tp / (tp + fn)}\tprecision = {recall}"

        print(f"{relation_name}")
        print(f"\tPrecision: {round(precision, 4)}\tRecall: {round(recall, 4)}\tF1: {round(f1, 4)}")
        print(f"\tTP: {tp}\tFP: {fp}\tTN: {tn}\tFN: {fn}")
        # print(f"precision_{relation_name}: {round(precision, 4)}\nrecall_{relation_name}: {round(recall, 4)}\nf1_{relation_name}: {round(f1, 4)}\n")

    print(f"The following relations are not present in the Gold Standard: {', '.join(relations_not_in_gs)}")
    print(f"The following relations are not present in the Predictions: {', '.join(relations_not_in_pred)}")

    print("\nGlobal results across all DrugProt relations (micro-average)")

    y_true_flattened = sum(y_true, [])
    y_pred_flattened = sum(y_pred, [])
    tn, fp, fn, tp = confusion_matrix(y_true_flattened, y_pred_flattened).ravel()

    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    # assert tp / (tp + fp + 1e-10) == precision, f"tp / (tp + fp) = {tp / (tp + fp)}\tprecision = {precision}"
    # assert tp / (tp + fn + 1e-10) == recall, f"tp / (tp + fp) = {tp / (tp + fn)}\tprecision = {recall}"

    print("Micro scores")
    print(f"\tPrecision: {round(precision, 4)}\tRecall: {round(recall, 4)}\tF1: {round(f1, 4)}")
    print(f"\tTP: {tp}\tFP: {fp}\tTN: {tn}\tFN: {fn}")

    # print(f"\np_micro: {round(precision, 4)}\nr_micro: {round(recall, 4)}\nf1_micro: {round(f1, 4)}\n")
