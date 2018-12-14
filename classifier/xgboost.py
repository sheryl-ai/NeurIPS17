#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import metrics
import xgboost as xgb
import pandas as pd
import numpy as np


# train on nips stage 1 data, test on stage1_test data using xgboost
def xgb_stage1_test_valid(train_x, train_y, feat_name=None):
    if feat_name is None:
        print('Not able to store stage1_test train/valid results. Empty feature name.')

    # split validation set
    samples = np.take(train_x, list(range(3321)), axis=0)
    labels = np.take(train_y, list(range(3321)))
    stage1_test_data = np.take(train_x, list(range(3321, 3689)), axis=0)
    stage1_test_lbl = np.take(train_y, list(range(3321, 3689)))

    # initialize parameters for xgboost
    params = {
        'eta': 0.03333,
        'max_depth': 4,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'silent': True,
        'tree_method': 'exact'
    }

    watchlist = [(xgb.DMatrix(samples, labels), 'train'), (xgb.DMatrix(stage1_test_data, stage1_test_lbl), 'valid')]

    print('Training xgboost on 3321 training set...')
    model = xgb.train(params, xgb.DMatrix(samples, labels), 1000, watchlist, verbose_eval=20, early_stopping_rounds=100)
    train_score = metrics.log_loss(labels, model.predict(xgb.DMatrix(samples), ntree_limit=model.best_ntree_limit + 80), labels=list(range(9)))

    print('Testing xgboost on 368 validation set...')
    valid_score = metrics.log_loss(stage1_test_lbl, model.predict(xgb.DMatrix(stage1_test_data), ntree_limit=model.best_ntree_limit) + 80, labels=list(range(9)))
    print('train_score =', train_score, ' valid_score =', valid_score)

    if feat_name is not None:
        # save stage1_test result
        pred1 = model.predict(xgb.DMatrix(samples), ntree_limit=model.best_ntree_limit + 80)
        kfold_submission1 = pd.DataFrame(pred1, columns=['class' + str(c + 1) for c in range(9)])
        kfold_submission1['ID'] = list(range(3321))
        try:
            kfold_submission1.to_csv('data/stage1_test_368/train.xgb.' + feat_name + '.csv', index=False)
        except FileNotFoundError:
            print(feat_name)
            kfold_submission1.to_csv('data/stage1_test_368/train.xgb.feat_name.csv', index=False)

        pred2 = model.predict(xgb.DMatrix(stage1_test_data), ntree_limit=model.best_ntree_limit + 80)
        kfold_submission2 = pd.DataFrame(pred2, columns=['class' + str(c + 1) for c in range(9)])
        kfold_submission2['ID'] = list(range(3321, 3689))
        try:
            kfold_submission2.to_csv('data/stage1_test_368/valid.xgb.' + feat_name + '.csv', index=False)
        except FileNotFoundError:
            print(feat_name)
            kfold_submission2.to_csv('data/stage1_test_368/valid.xgb.feat_name.csv', index=False)
    return model.best_ntree_limit, train_score, valid_score


# n fold cross validation using xgboost
def xgb_nfold_cv(k, feat_name, x_train, x_valid, y_train, y_valid, train_index, test_index, test_samples, nfold=5):
    # initialize parameters for xgboost
    params = {
        'eta': 0.03333,
        'max_depth': 4,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': k,
        'silent': True,
        'tree_method': 'exact'
    }

    watchlist = [(xgb.DMatrix(x_train, y_train), 'train'), (xgb.DMatrix(x_valid, y_valid), 'valid')]

    print('Training GBDT on ' + str(nfold) + ' fold cross validation...')
    model = xgb.train(params, xgb.DMatrix(x_train, y_train), 1000, watchlist, verbose_eval=20, early_stopping_rounds=100)
    score = metrics.log_loss(y_valid, model.predict(xgb.DMatrix(x_valid), ntree_limit=model.best_ntree_limit), labels=list(range(9)))
    print('logloss on %d fold: ' % (k + 1))
    print(score)

    print('Testing GBDT on ' + str(nfold) + ' fold cross validation...')
    pred = model.predict(xgb.DMatrix(test_samples), ntree_limit=model.best_ntree_limit + 80)

    # save results
    pred1 = model.predict(xgb.DMatrix(x_train), ntree_limit=model.best_ntree_limit + 80)
    kfold_submission1 = pd.DataFrame(pred1, columns=['class' + str(c + 1) for c in range(9)])
    kfold_submission1['ID'] = train_index
    try:
        kfold_submission1.to_csv('data/' + str(nfold) + 'fold_cv/train.xgb.' + feat_name + '.fold_' + str(k) + '.csv', index=False)
    except FileNotFoundError:
        print(feat_name)
        kfold_submission1.to_csv('data/' + str(nfold) + '0fold_cv/train.xgb.feat_name.fold_' + str(k) + '.csv', index=False)

    pred2 = model.predict(xgb.DMatrix(x_valid), ntree_limit=model.best_ntree_limit + 80)
    kfold_submission2 = pd.DataFrame(pred2, columns=['class' + str(c + 1) for c in range(9)])
    kfold_submission2['ID'] = test_index
    try:
        kfold_submission2.to_csv('data/' + str(nfold) + 'fold_cv/valid.xgb.' + feat_name + '.fold_' + str(k) + '.csv', index=False)
    except FileNotFoundError:
        print(feat_name)
        kfold_submission2.to_csv('data/' + str(nfold) + 'fold_cv/valid.xgb.feat_name.fold_' + str(k) + '.csv', index=False)

    return pred, score
