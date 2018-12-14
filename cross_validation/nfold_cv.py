#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pickle

from classifier.xgboost import xgb_nfold_cv
from classifier.lightgbm import lgb_nfold_cv
from classifier.svc import svc_nfold_cv
from classifier.random_forest import rf_nfold_cv
from classifier.logistic_regression import lr_nfold_cv
from classifier.multi_layer_perceptron import mlp_nfold_cv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit


def load_kfold(nfold=5):
    with open('data/intermediate/' + str(nfold) + 'fold.map.pkl', 'rb') as f:
        kfold = pickle.load(f)
    return kfold


def get_cvsplitter(k, kfold, train_id):  # given kfold index k
    kfold_train_id = list()
    kfold_valid_id = list()
    for id in train_id:
        if kfold[id] == k:
            kfold_valid_id.append(id)
        else:
            kfold_train_id.append(id)
    return kfold_train_id, kfold_valid_id


def select_classifier(k, feat_name, tree_feat_dict, train_array, train_y, kfold_train_id, valid_array, valid_y, kfold_valid_id, test_feats, nfold=5, classifier='xgb', flag='single'):
    if classifier == 'xgb':
        pred, score = xgb_nfold_cv(k, feat_name, train_array, valid_array, train_y, valid_y, kfold_train_id, kfold_valid_id, test_feats, nfold)
    elif classifier == 'lgb':
        pred, score = lgb_nfold_cv(k, feat_name, train_array, valid_array, train_y, valid_y, kfold_train_id, kfold_valid_id, test_feats, nfold)
    elif classifier == 'rf':
        if flag == 'single':
            n_tree = tree_feat_dict[feat_name]
        else:
            n_tree = 0
            features = feat_name.split('+')
            for feat in features:
                n_tree += tree_feat_dict[feat]
                n_tree /= len(features)
        pred, score = rf_nfold_cv(k, feat_name, int(n_tree), train_array, valid_array, train_y, valid_y, kfold_train_id, kfold_valid_id, test_feats, nfold)
    else:
        scaler = MinMaxScaler()
        train_array = scaler.fit_transform(train_array)
        valid_array = scaler.fit_transform(valid_array)
        test_feats = scaler.fit_transform(test_feats)
        if classifier == 'lr':
            pred, score = lr_nfold_cv(k, feat_name, train_array, valid_array, train_y, valid_y, kfold_train_id, kfold_valid_id, test_feats, nfold)
        elif classifier == 'mlp':
            pred, score = mlp_nfold_cv(k, feat_name, train_array, valid_array, train_y, valid_y, kfold_train_id, kfold_valid_id, test_feats, nfold)
        elif classifier == 'svc':
            pred, score = svc_nfold_cv(k, feat_name, train_array, valid_array, train_y, valid_y, kfold_train_id, kfold_valid_id, test_feats, nfold)
    return pred, score


def rand_nfold_cv(feat_name, tree_feat_dict, train_feats, train_y, test_feats, test_index, nfold=5, classifier='xgb', flag='single'):
    sss = StratifiedShuffleSplit(n_splits=nfold, test_size=0.1)

    preds = 0
    logloss = 0
    scores = []
    k = 0

    # get random 5fold cross validation split
    for train_index, valid_index in sss.split(train_feats, train_y):
        x_train, x_valid = train_feats[train_index], train_feats[valid_index]
        y_train, y_valid = train_y[train_index], train_y[valid_index]

        pred, score = select_classifier(k, feat_name, tree_feat_dict, x_train, train_y, train_index, x_valid, y_valid, valid_index, test_feats, nfold, classifier, flag)
        preds += pred
        logloss += score
        scores.append(score)
        k += 1

    preds /= nfold
    logloss /= nfold

    # save prediction results
    res = pd.DataFrame(preds, columns=['class' + str(c + 1) for c in range(9)])
    res['ID'] = test_index

    try:
        res.to_csv('data/truth/' + str(nfold) + 'fold_cv/' + flag + '/submission.' + classifier + '.' + feat_name + '.csv', index=False)
    except FileNotFoundError:
        print(feat_name)
        res.to_csv('data/truth/' + str(nfold) + 'fold_cv/' + flag + '/submission.' + classifier + '.feat_name.csv', index=False)

    del k
    print('The Logloss result on ' + str(nfold) + ' fold CV is: ', logloss)
    with open('single_feature_' + str(nfold) + 'fold.tsv', 'a') as f:
        f.write(feat_name + '\t')
        for k in range(nfold):
            if k == nfold - 1:
                f.write(str(scores[k]) + '\n')
            else:
                f.write(str(scores[k]) + '\t')
