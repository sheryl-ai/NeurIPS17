#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import combinations

from ensemble.feature_fusion import *


# load and stack probability generated previously
def _load_proba(train_id, train_y, k, kfold, comb):
    if k == -1 and kfold is None:
        train_proba = []
        valid_proba = []

        for feat in comb:
            train = pd.read_csv('data/stage1_test_368/train.' + feat + '.csv').values[:, 0:9]
            valid = pd.read_csv('data/stage1_test_368/valid.' + feat + '.csv').values[:, 0:9]

            train_proba += [train]
            valid_proba += [valid]

        train_data = np.vstack([train_proba])
        valid_data = np.vstack([valid_proba])

        return train_data, valid_data

    else:   # for 5 fold cross-validation data
        kfold_train_id, kfold_valid_id = get_cvsplitter(k, kfold, train_id)
        train_lbl = train_y[kfold_train_id]
        valid_lbl = train_y[kfold_valid_id]

        train_proba = []
        valid_proba = []

        for feat in comb:
            train = pd.read_csv('data/5fold_cv/train.' + feat + '.fold_' + str(k) + '.csv').values[:, 0:9]
            valid = pd.read_csv('data/5fold_cv/valid.' + feat + '.fold_' + str(k) + '.csv').values[:, 0:9]

            train_proba += [train]
            valid_proba += [valid]

        train_data = np.vstack([train_proba])
        valid_data = np.vstack([valid_proba])

        return train_data, train_lbl, valid_data, valid_lbl


def _normalize(data):
    norm = np.sum(data, axis=1)
    for i in range(len(norm)):
        data[i, :] /= norm[i]
    return data


def _load_valid_N_y(valid_label):
    valid_N_y = np.zeros(9, dtype=np.uint16)
    lbl_cnt = np.unique(valid_label, return_counts=True)
    for i in range(9):
        valid_N_y[i] = lbl_cnt[1][i]
    return valid_N_y


def _accuracy_new(data_valid, label_valid, valid_N_y, comb, solos, num):
    """ x is an array of shape nsamples, nclassifiers
    k is an array of size nclassifiers
    this function return the accuracy of the linear combination
    with k coefficients
    """
    multiple_vacc = np.zeros([num, 9])
    N = len(label_valid)
    _, m, n = data_valid.shape
    for c in comb:
        i = solos.index(c)
        j = comb.index(c)
        pred = data_valid[i, :, :]

        loss = np.zeros(9)
        for n in range(N):
            y = label_valid[n]
            loss[y] += np.log(pred[n, y])
        logloss = [-1.0 * loss[i] / valid_N_y[i] if valid_N_y[i] != 0 else 0 for i in range(9)]
        multiple_vacc[j, :] = np.exp(-np.array(logloss))

    multiple_vacc = multiple_vacc / np.sum(multiple_vacc, axis=0)
    return multiple_vacc


def _accuracy_old(data_valid, label_valid, comb, solos):
    multiple_vacc = []
    N = len(label_valid)
    _, m, n = data_valid.shape

    for c in comb:
        i = solos.index(c)
        pred = data_valid[i, :, :]
        loss = 0
        for n in range(N):
            y = label_valid[n]
            loss += np.log(pred[n, y])
        logloss = -1.0 / N * loss
        multiple_vacc.append(np.exp(-logloss))

    multiple_vacc = multiple_vacc / np.sum(multiple_vacc)
    return multiple_vacc


def get_logloss(weights, data, label, comb, solo, method='brute_force'):
    """ computes the weigths of each _ensemble
    according to the contribution of each model
    on the valid set """
    _, m, n = data.shape
    pred_comb = np.zeros([m, n])
    for c in comb:
        if method == 'accuracy_new':
            pred_comb += weights[comb.index(c), :] * data[solo.index(c), :, :]
        elif method == 'brute_force' or method == 'accuracy_old':
            pred_comb += weights[comb.index(c)] * data[solo.index(c), :, :]

    loss = 0
    N = len(label)
    for i in range(N):
        loss += np.log(pred_comb[i, label[i]])
    logloss = -1.0 / N * loss
    return logloss


def get_nfold_results_ensemble(method='brute_force', num=3, nfold=5):
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trains = [file for file in os.listdir('data/5fold_cv/') if file.startswith('train')]
    valids = [file for file in os.listdir('data/5fold_cv/') if file.startswith('valid')]
    results = [file for file in os.listdir('data/5fold_cv/') if file.startswith('submission')]

    if len(results) == len(trains) / nfold == len(valids) / nfold:
        comb = list(combinations([c.split('.')[1] + '.' + c.split('.')[2] for c in results], num))
        solo = list(set([item.split('.')[1] + '.' + item.split('.')[2] for item in results]))
        ens_weights = dict()
        ens_logloss = dict()
        for c in comb:
            if method == 'accuracy_new':
                ens_weights['.'.join(c)] = np.zeros([num, 9], dtype='float32')
            elif method == 'accuracy_old' or method == 'brute_force':
                ens_weights['.'.join(c)] = np.zeros([num], dtype='float32')
            ens_logloss['.'.join(c)] = 0

        train_x, train_y, test_x, test_index = load_data(flag='stage2')
        train_y -= 1
        kfold = load_kfold(nfold=nfold)

        for k in range(nfold):
            print()
            print('Importing fold', k + 1, ' results...')
            train_data, train_lbl, valid_data, valid_lbl = _load_proba(train_x['ID'].tolist(), train_y, k, kfold, solo)

            if method == 'accuracy_new':
                valid_N_y = _load_valid_N_y(valid_lbl)

            for c in comb:
                logloss = OrderedDict()
                if method == 'brute_force':
                    if num == 2:
                        for w in weights:
                            logloss[str(w) + '_' + str(1 - w)] = get_logloss([w, 1 - w], valid_data, valid_lbl, c, solo,
                                                                             method)
                    elif num == 3:
                        wgts = [(x, y, 1 - x - y) for y in weights for x in weights if 1 - x - y > 0.09]
                        for w in wgts:
                            logloss['_'.join([str(i) for i in w])] = get_logloss(list(w), valid_data, valid_lbl, c,
                                                                                 solo, method)

                elif method == 'accuracy_old':
                    w = _accuracy_old(valid_data, valid_lbl, c, solo)
                    logloss['_'.join([str(i) for i in w])] = get_logloss(w, valid_data, valid_lbl, c, solo, method)

                elif method == 'accuracy_new':
                    w = _accuracy_new(valid_data, valid_lbl, valid_N_y, c, solo, num)
                    logloss['_'.join([str(i) for i in w.flatten()])] = get_logloss(w, valid_data, valid_lbl, c, solo,
                                                                                   method)

                # find the optimal weights for each method
                opt_pair = min(logloss, key=logloss.get)
                ens_logloss['.'.join(c)] += logloss[opt_pair]
                print('Feature combination:', c)

                if method == 'accuracy_new':
                    ens_weights['.'.join(c)] += np.array([float(x) for x in opt_pair.split('_')]).reshape(num, 9)
                    print('Optimal weights:', '\n', np.array([float(x) for x in opt_pair.split('_')]).reshape(num, 9))
                elif method == 'accuracy_old' or method == 'brute_force':
                    ens_weights['.'.join(c)] += np.array([float(x) for x in opt_pair.split('_')])
                    print('Optimal weights:', tuple([float(x) for x in opt_pair.split('_')]))
                print('Optimal validation set log loss:', logloss[opt_pair])
                print()

        # find the optimal method
        print()
        print('*' * 50, 'ensemble Results', '*' * 50)
        print('The average log loss of the combinations:')
        for c in comb:
            print(ens_logloss['.'.join(c)] / nfold)

        idx = np.argmin(np.array([ens_logloss['.'.join(c)] / nfold for c in comb]))
        opt_comb = comb[idx]
        opt_weights = ens_weights['.'.join(opt_comb)] / nfold
        opt_logloss = np.min(np.array([ens_logloss['.'.join(c)] / nfold for c in comb]))
        print('Feature combination:', opt_comb)
        if method == 'accuracy_new':
            print('Optimal weights:', '\n', opt_weights)
        elif method == 'accuracy_old' or method == 'brute_force':
            print('Optimal weights:', tuple(opt_weights))
        print('Optimal validation set log loss:', opt_logloss)

    else:
        raise ValueError('Incomplete train/valid/submission files.')


def get_stage1_test_results_ensemble(method='brute_force', num=3):
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trains = [file for file in os.listdir('data/stage1_test_368/') if file.startswith('train')]
    valids = [file for file in os.listdir('data/stage1_test_368/') if file.startswith('valid')]

    if len(trains) == len(valids):
        comb = list(combinations([c.split('.')[1] + '.' + c.split('.')[2] for c in trains], num))
        solo = list(set([item.split('.')[1] + '.' + item.split('.')[2] for item in trains]))
        ens_weights = dict()
        ens_logloss = dict()
        for c in comb:
            if method == 'accuracy_new':
                ens_weights['.'.join(c)] = np.zeros([num, 9], dtype='float32')
            elif method == 'accuracy_old' or method == 'brute_force':
                ens_weights['.'.join(c)] = np.zeros([num], dtype='float32')
            ens_logloss['.'.join(c)] = 0

        train_x, train_y, test_x, test_index = load_data(flag='stage2')
        train_y -= 1
        train_data, valid_data = _load_proba(train_x['ID'].tolist(), train_y, -1, None, solo)
        valid_lbl = np.take(train_y, list(range(3321, 3689)))

        if method == 'accuracy_new':
            valid_N_y = _load_valid_N_y(valid_lbl)

        for c in comb:
            logloss = OrderedDict()
            if method == 'brute_force':
                if num == 2:
                    for w in weights:
                        logloss[str(w) + '_' + str(1 - w)] = get_logloss([w, 1 - w], valid_data, valid_lbl, c, solo,
                                                                         method)
                elif num == 3:
                    wgts = [(x, y, 1 - x - y) for y in weights for x in weights if 1 - x - y > 0.09]
                    for w in wgts:
                        logloss['_'.join([str(i) for i in w])] = get_logloss(list(w), valid_data, valid_lbl, c, solo,
                                                                             method)

            elif method == 'accuracy_old':
                w = _accuracy_old(valid_data, valid_lbl, c, solo)
                logloss['_'.join([str(i) for i in w])] = get_logloss(w, valid_data, valid_lbl, c, solo, method)

            elif method == 'accuracy_new':
                w = _accuracy_new(valid_data, valid_lbl, valid_N_y, c, solo, num)
                logloss['_'.join([str(i) for i in w.flatten()])] = get_logloss(w, valid_data, valid_lbl, c, solo,
                                                                               method)

            # find the optimal weights for each method
            opt_pair = min(logloss, key=logloss.get)
            ens_logloss['.'.join(c)] = logloss[opt_pair]
            print('Feature combination:', c)

            if method == 'accuracy_new':
                ens_weights['.'.join(c)] = np.array([float(x) for x in opt_pair.split('_')]).reshape(num, 9)
                print('Optimal weights:', '\n', np.array([float(x) for x in opt_pair.split('_')]).reshape(num, 9))
            elif method == 'accuracy_old' or method == 'brute_force':
                ens_weights['.'.join(c)] = np.array([float(x) for x in opt_pair.split('_')])
                print('Optimal weights:', tuple([float(x) for x in opt_pair.split('_')]))
            print('Optimal validation set log loss:', logloss[opt_pair])
            print()

        print()
        print('*' * 50, 'ensemble Results', '*' * 50)
        # find the optimal method
        idx = np.argmin(np.array([ens_logloss['.'.join(c)] for c in comb]))
        opt_comb = comb[idx]
        opt_weights = ens_weights['.'.join(opt_comb)]
        opt_logloss = np.min(np.array([ens_logloss['.'.join(c)] for c in comb]))
        print('Feature combination:', opt_comb)
        if method == 'accuracy_new':
            print('Optimal weights:', '\n', opt_weights)
        elif method == 'accuracy_old' or method == 'brute_force':
            print('Optimal weights:', tuple(opt_weights))
        print('Optimal validation set log loss:', opt_logloss)

    else:
        raise ValueError('Incomplete train/valid/submission files.')
