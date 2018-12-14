#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

from helper import *
from cross_validation.nfold_cv import *


# ensemble features by concatenating them
def combine_feat(comb):
    for feat_name in comb:
        train_feat, test_feat = load_feature(feat_name)

        if 'train' in locals() and 'test' in locals():
            train = np.concatenate((train, train_feat), axis=1)
            test = np.concatenate((test, test_feat), axis=1)
        else:
            train = copy.deepcopy(train_feat)
            test = copy.deepcopy(test_feat)
    print('+'.join(comb) + ': train_feat.shape', train.shape, ' test_feat.shape', test.shape)
    return train, test


def get_feature_fusion(fusions, train_y, train_id, test_index, classifier='xgb'):
    for comb in fusions:
        train_feat, test_feat = combine_feat(comb)

        # apply random n fold cross-validation for ensembled feature
        rand_nfold_cv('+'.join(comb), train_feat, train_y, train_id, test_feat, test_index, classifier=classifier, flag='combine')
