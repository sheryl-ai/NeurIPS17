#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature.name_mining import *
from feature.document_mining import *
from feature.relation_mining import *

from classifier.xgboost import *
from classifier.random_forest import *
from classifier.lightgbm import *
from classifier.logistic_regression import *
from classifier.svc import *
from classifier.multi_layer_perceptron import *

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def load_data(flag='stage1'):
    print('Loading data...')
    train_variant = pd.read_csv("data/training_variants")
    train_text = pd.read_csv("data/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID", "Text"])
    test_variant = pd.read_csv("data/test_variants")
    test_text = pd.read_csv("data/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID", "Text"])
    if flag == 'stage1':
        train = pd.merge(train_variant, train_text, how='left', on='ID')
        train_y = train['Class'].values
        train_x = train.drop('Class', axis=1)

        test_x = pd.merge(test_variant, test_text, how='left', on='ID')
        test_index = test_x['ID'].values
    else:
        stage1_test = pd.read_csv('data/stage1_solution_filtered.csv')
        stage1_test_variant = test_variant.loc[stage1_test['ID']]
        stage1_test_variant['ID'] = [id + 3321 for id in range(stage1_test.shape[0])]
        labels = list(stage1_test[stage1_test != 0].drop(['ID'], axis=1).stack().index)
        stage1_test_variant['Class'] = pd.Series(data=[int(val[1][-1]) for val in labels], index=stage1_test['ID'])
        stage1_test_text = test_text.loc[stage1_test['ID']]

        stage2_variant = pd.read_csv('data/stage2_test_variants.csv')
        stage2_text = pd.read_csv('data/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID", "Text"])

        train_variant = pd.concat((train_variant, stage1_test_variant), axis=0, ignore_index=True)
        train_text = pd.concat((train_text, stage1_test_text), axis=0, ignore_index=True)

        train = pd.concat((train_variant, train_text['Text']), axis=1, ignore_index=True)
        train.columns = ['ID', 'Gene', 'Variation', 'Class', 'Text']

        train_y = train['Class'].values
        train_x = train.drop('Class', axis=1)
        test_index = pd.read_csv('data/stage2_sample_submission.csv')['ID']
        if flag == 'stage2':
            test_x = pd.merge(stage2_variant, stage2_text, how='left', on='ID')
            test_x['ID'] = list(range(stage2_variant.shape[0]))
            test_index = test_index.values
        elif flag == 'truth':
            truth = get_true_label()
            id = truth['ID'].map(lambda x: x-1).tolist()
            stage2_variant = stage2_variant.loc[id]
            stage2_text = stage2_text.loc[id]

            test_x = pd.merge(stage2_variant, stage2_text, how='left', on='ID')
            test_x['ID'] = list(range(stage2_variant.shape[0]))
            test_index = test_index.loc[id].values

    print('train_x.shape', train_x.shape, ' test_x.shape', test_x.shape)
    return train_x, train_y, test_x, test_index


def get_feature(train_x, test_x, train_y):
    # name mining
    get_entity_name_feats(train_x, test_x)
    word2vec_feats(train_x, test_x, 100)
    get_gene_var_distribute(train_x, test_x, train_y)

    # document mining
    doc2vec_feats(train_x, test_x, 150, 250)

    get_text_mining_feats(train_x, test_x)

    pos_tagging_feats(train_x, test_x, train_y)
    pos_tagging_nmf()
    bioentity_feats(train_x, test_x)
    pubmed_feats(train_x, test_x)

    gene_var_text_len(train_x, test_x)
    text_id(train_x, test_x)

    # relation mining
    get_relation_mining_feats(train_x, test_x)


def load_feature(feat_name):
    print()
    print('Import ' + feat_name + ' feature...')
    train_feats = pd.read_csv('data/features/' + feat_name + '_train.csv', header=None).values
    test_feats = pd.read_csv('data/features/' + feat_name + '_test.csv', header=None).values
    print('train_feats.shape', train_feats.shape, ' test_feats.shape', test_feats.shape)
    return train_feats, test_feats


def get_classifier(feat_name, train_x, train_y):
    n_tree, train_score_xgb, valid_score_xgb = xgb_stage1_test_valid(train_x, train_y)    # xgboost
    train_score_lgb, valid_score_lgb = lgb_stage1_test_valid(train_x, train_y)            # lightgbm
    train_score_rf, valid_score_rf = rf_stage1_test_valid(train_x, train_y, n_tree)       # random forest
    train_score_lr, valid_score_lr = lr_stage1_test_valid(train_x, train_y)               # logistic regression
    train_score_svc, valid_score_svc = svc_stage1_test_valid(train_x, train_y)            # c-support vector classification
    train_score_mlp, valid_score_mlp = mlp_stage1_test_valid(train_x, train_y)            # multi-layer perceptron classifier

    with open('feature_combination_validation.tsv', 'a') as f:
        f.write(feat_name + '\t' + str(train_score_xgb) + '\t' + str(valid_score_xgb) + '\t' +
                str(train_score_lgb) + '\t' + str(valid_score_lgb) + '\t' +
                str(train_score_rf) + '\t' + str(valid_score_rf) + '\t' +
                str(train_score_lr) + '\t' + str(valid_score_lr) + '\t' +
                str(train_score_svc) + '\t' + str(valid_score_svc) + '\t' +
                str(train_score_mlp) + '\t' + str(valid_score_mlp) + '\n')


def get_true_label():   # load true label given by Kaggle website
    df = pd.read_csv('data/stage_2_private_solution.csv')
    label = list(df[df != 0].drop(['ID'], axis=1).stack().index)
    truth = pd.DataFrame({'ID': df['ID'].tolist(), 'Class': [int(val[1][-1]) for val in label]})[['ID', 'Class']]
    return truth
