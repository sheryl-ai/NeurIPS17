#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import lightgbm as lgb
import pandas as pd
from sklearn import metrics
import numpy as np


# ltrain on nips stage 1 data, test on stage1_test data using lightgbm
def lgb_stage1_test_valid(train_x, train_y, feat_name=None):
    if feat_name is None:
        print('Not able to store stage1_test train/valid results. Empty feature name.')

    # split validation set
    samples = np.take(train_x, list(range(3321)), axis=0)
    labels = np.take(train_y, list(range(3321)))
    stage1_test_data = np.take(train_x, list(range(3321, 3689)), axis=0)
    stage1_test_lbl = np.take(train_y, list(range(3321, 3689)))

    # initialize parameters for lightgbm
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 9,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'lambda_l1': 1.0,
        'verbose': 0
    }
    print('Training lightgbm on 3321 training set...')
    lgb_train = lgb.Dataset(samples, labels)
    model = lgb.train(params, lgb_train, num_boost_round=1000, verbose_eval=20)
    train_score = metrics.log_loss(labels, model.predict(samples, num_iteration=model.best_iteration))

    print('Testing lightgbm on 368 validation set...')
    valid_score = metrics.log_loss(stage1_test_lbl, model.predict(stage1_test_data, num_iteration=model.best_iteration))
    print('train_score =', train_score, ' valid_score =', valid_score)

    if feat_name is not None:
        # save stage1_test result
        pred1 = model.predict(samples, num_iteration=model.best_iteration)
        kfold_submission1 = pd.DataFrame(pred1, columns=['class' + str(c + 1) for c in range(9)])
        kfold_submission1['ID'] = list(range(3321))
        try:
            kfold_submission1.to_csv('data/stage1_test_368/train.lgb.' + feat_name + '.csv', index=False)
        except FileNotFoundError:
            print(feat_name)
            kfold_submission1.to_csv('data/stage1_test_368/train.lgb.feat_name.csv', index=False)

        pred2 = model.predict(stage1_test_data, num_iteration=model.best_iteration)
        kfold_submission2 = pd.DataFrame(pred2, columns=['class' + str(c + 1) for c in range(9)])
        kfold_submission2['ID'] = list(range(3321, 3689))
        try:
            kfold_submission2.to_csv('data/stage1_test_368/valid.lgb.' + feat_name + '.csv', index=False)
        except FileNotFoundError:
            print(feat_name)
            kfold_submission2.to_csv('data/stage1_test_368/valid.lgb.feat_name.csv', index=False)
    return train_score, valid_score


# n fold cross validation using lightgbm
def lgb_nfold_cv(k, feat_name, x_train, x_valid, y_train, y_valid, train_index, test_index, test_samples, nfold=5):
    # initialize parameters for lightgbm
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 9,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'lambda_l1': 1.0,
        'verbose': -1
    }
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

    print('Training LightGBM on ' + str(nfold) + ' fold cross validation...')
    model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, verbose_eval=20, early_stopping_rounds=100)
    score = metrics.log_loss(y_valid, model.predict(x_valid, num_iteration=model.best_iteration))
    print('logloss on %d fold: ' % (k + 1))
    print(score)

    print('Testing LightGBM on ' + str(nfold) + ' fold cross validation...')
    pred = model.predict(test_samples, num_iteration=model.best_iteration)

    pred1 = model.predict(x_train, num_iteration=model.best_iteration)
    kfold_submission1 = pd.DataFrame(pred1, columns=['class' + str(c + 1) for c in range(9)])
    kfold_submission1['ID'] = train_index
    try:
        kfold_submission1.to_csv('data/' + str(nfold) + 'fold_cv/train.lgb.' + feat_name + '.fold_' + str(k) + '.csv', index=False)
    except FileNotFoundError:
        print(feat_name)
        kfold_submission1.to_csv('data/' + str(nfold) + 'fold_cv/train.1gb.feat_name.fold_' + str(k) + '.csv', index=False)

    pred2 = model.predict(x_valid, num_iteration=model.best_iteration)
    kfold_submission2 = pd.DataFrame(pred2, columns=['class' + str(c + 1) for c in range(9)])
    kfold_submission2['ID'] = test_index
    try:
        kfold_submission2.to_csv('data/' + str(nfold) + 'fold_cv/valid.lgb.' + feat_name + '.fold_' + str(k) + '.csv', index=False)
    except FileNotFoundError:
        print(feat_name)
        kfold_submission2.to_csv('data/' + str(nfold) + 'fold_cv/valid.lgb.feat_name.fold_' + str(k) + '.csv', index=False)

    return pred, score
