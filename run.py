#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ensemble.result_ensemble import *
from cross_validation.nfold_cv import *
from helper import *

import pickle

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def classification(train_x, test_x, train_y, tree_feat_dict):
    # learn features from gene/variation name, document and name-text relation
    get_feature(train_x, test_x, train_y)

    for feat_name in tree_feat_dict.keys():
        # load features
        train_feats, test_feats = load_feature(feat_name)

        # multi-classification for each feature
        get_classifier(feat_name, train_feats, train_y)

        # five fold cross validation
        rand_nfold_cv(feat_name, tree_feat_dict, train_feats, train_y, test_feats, test_x['ID'].values, nfold=5, classifier='xgb', flag='single')


def ensemble(train_x, train_y, test_index):
    fusions = [
        ['gene_var_distribute', 'unique_dict_all_doc_rep', 'unique_dict_all_nmf', 'text_doc_svd', 'word2vec100'],
        ['gene_var_distribute', 'unique_dict_all_doc_rep', 'unique_dict_all_nmf', 'sent_tfidf_word_char_svd', 'word2vec100'],
        ['gene_var_distribute', 'unique_dict_all_nmf', 'gene_doc_rep', 'text_doc_svd', 'word2vec100']
    ]

    # feature ensemble
    for c in ['xgb', 'lgb', 'rf', 'lr', 'svc', 'mlp']:
        get_feature_fusion(fusions, train_y, train_x['ID'].values, test_index, classifier=c)

    # result ensemble
    # get_nfold_results_ensemble(method='brute_force', num=3)
    get_nfold_results_ensemble(method='accuracy_old', num=3)
    # get_nfold_results_ensemble(method='accuracy_new', num=3)


def main():
    # load pre-defined number of trees for random forest algorithm
    with open('data/pre_define/tree_feat_dict.pkl', 'rb') as f:
        tree_feat_dict = pickle.load(f)

    # load data given by Kaggle website
    train_x, train_y, test_x, test_index = load_data(flag='truth')
    train_y -= 1

    classification(train_x, test_x, train_y, tree_feat_dict)
    ensemble(train_x, train_y, test_index)


if __name__ == '__main__':
    main()
