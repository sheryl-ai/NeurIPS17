#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from sklearn import metrics
import numpy as np
import pandas as pd
import argparse


def normalize(data):
    norm = np.sum(data, axis=1)
    for i in range(len(norm)):
        data[i, :] /= norm[i]
    return data

def load_weights(num, train_label=None):
    with open('data/feat_wgt_dict.pkl', 'rb') as f:
        feat_wgt_dict = pickle.load(f)

    wgt_proba = np.zeros((num, 9))
    for item in feat_wgt_dict.items():
        # merge validation set from five fold cross validation
        valid0 = pd.read_csv('data/5fold_cv/valid.' + item[0] + '.fold_0.csv')
        valid1 = pd.read_csv('data/5fold_cv/valid.' + item[0] + '.fold_1.csv')
        valid2 = pd.read_csv('data/5fold_cv/valid.' + item[0] + '.fold_2.csv')
        valid3 = pd.read_csv('data/5fold_cv/valid.' + item[0] + '.fold_3.csv')
        valid4 = pd.read_csv('data/5fold_cv/valid.' + item[0] + '.fold_4.csv')
        valid = pd.concat([valid0, valid1, valid2, valid3, valid4], axis=0)
        proba = valid.sort_values('ID').reset_index().drop(['index', 'ID'], axis=1).values

        wgt_proba += proba * item[1]

    wgt_proba = normalize(wgt_proba)

    if train_label is not None:
        score = metrics.log_loss(train_label, wgt_proba)
        print()
        print('train + valid log loss:', score)
    return wgt_proba


def prediction_demo(stage1_x, stage1_label, gene, variation):
    wgt_proba = load_weights(len(stage1_x), None)
    for idx, row in stage1_x.iterrows():
        if gene == row['Gene'] and variation == row['Variation']:
            pred = wgt_proba[row['ID']]
            print('ID:', row['ID'], 'Gene:', row['Gene'], 'Variation:', row['Variation'])

            gt = np.zeros(9)
            gt[stage1_label[row['ID']]-1] = 1
            print('{:^10} {:^20} {:<20}'.format('class', 'prediction', 'groundtruth'))
            for cls in range(9):
                print('{:^10} {:<25} {:<3}'.format(cls+1, pred[cls], gt[cls]))
            return pred

    if 'pred' not in locals():
        print('No information for the given gene or variation. Please try again.')


if __name__ == '__main__':
    # load neurips stage1 train and test data
    stage1_variant = pd.read_csv('data/stage1_variants')
    stage2_variant = pd.read_csv('data/stage2_variants')

    # load stage1_solution data
    stage1_solution = pd.read_csv('data/stage1_solution_filtered.csv')
    stage1_solution_variant = stage2_variant.loc[stage1_solution['ID']]
    stage1_solution_variant['ID'] = [id + 3321 for id in range(stage1_solution.shape[0])]
    labels = list(stage1_solution[stage1_solution != 0].drop(['ID'], axis=1).stack().index)
    stage1_solution_variant['Class'] = pd.Series(data=[int(val[1][-1]) for val in labels], index=stage1_solution['ID'])

    # merge neurips stage1 and stage1_solution
    stage1 = pd.concat((stage1_variant, stage1_solution_variant), axis=0, ignore_index=True)
    stage1_label = stage1['Class'].values
    stage1_x = stage1.drop(['Class'], axis=1)

    parser = argparse.ArgumentParser(description='demo.py: get predicted probability for given (gene, variation) pair')
    parser.add_argument('-gene', help='gene name')
    parser.add_argument('-variation', help='vairation name')
    parser.print_help()
    args = parser.parse_args()
    print()

    # demo for prediction given (gene, variation)
    pred = prediction_demo(stage1_x, stage1_label, gene=args.gene, variation=args.variation)
