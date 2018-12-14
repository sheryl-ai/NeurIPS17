#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# train on nips stage 1 data, test on stage1_test data using c-support vector classification
def svc_stage1_test_valid(train_x, train_y):
    # preprocessing
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)

    # split validation set
    samples = np.take(train_x, list(range(3321)), axis=0)
    labels = np.take(train_y, list(range(3321)))
    stage1_test_data = np.take(train_x, list(range(3321, 3689)), axis=0)
    stage1_test_lbl = np.take(train_y, list(range(3321, 3689)))

    clf = SVC(probability=True, max_iter=1000)

    print('Training SVC on 3321 training set...')
    clf.fit(samples, labels)
    train_score = metrics.log_loss(labels, clf.predict_proba(samples))

    print('Testing SVC on 368 validation set...')
    valid_score = metrics.log_loss(stage1_test_lbl, clf.predict_proba(stage1_test_data))
    print('train_score =', train_score, ' valid_score =', valid_score)
    return train_score, valid_score


# n fold cross validation using c-support vector classification
def svc_nfold_cv(k, feat_name, x_train, x_valid, y_train, y_valid, train_index, test_index, test_samples, nfold=5):
    clf = SVC(probability=True, max_iter=-1)

    print('Training SVC on ' + str(nfold) + ' fold cross validation...')
    clf.fit(x_train, y_train)
    score = metrics.log_loss(y_valid, clf.predict_proba(x_valid))
    print('logloss on %d fold: ' % (k + 1))
    print(score)

    print('Testing SVC on ' + str(nfold) + ' fold cross validation...')
    pred = clf.predict_proba(test_samples)

    pred1 = clf.predict_proba(x_train)
    kfold_submission1 = pd.DataFrame(pred1, columns=['class' + str(c + 1) for c in range(9)])
    kfold_submission1['ID'] = train_index
    try:
        kfold_submission1.to_csv('data/' + str(nfold) + 'fold_cv/train.svc.' + feat_name + '.fold_' + str(k) + '.csv', index=False)
    except FileNotFoundError:
        print(feat_name)
        kfold_submission1.to_csv('data/' + str(nfold) + 'fold_cv/train.svc.feat_name.fold_' + str(k) + '.csv', index=False)

    pred2 = clf.predict_proba(x_valid)
    kfold_submission2 = pd.DataFrame(pred2, columns=['class' + str(c + 1) for c in range(9)])
    kfold_submission2['ID'] = test_index
    try:
        kfold_submission2.to_csv('data/' + str(nfold) + 'fold_cv/valid.svc.' + feat_name + '.fold_' + str(k) + '.csv', index=False)
    except FileNotFoundError:
        print(feat_name)
        kfold_submission2.to_csv('data/' + str(nfold) + 'fold_cv/valid.svc.feat_name.fold_' + str(k) + '.csv', index=False)

    return pred, score
