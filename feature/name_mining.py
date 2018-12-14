#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim.models as gs
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import os


def name_count_onehot(df_all):
    print('Counting character occurrences in each gene/variation name...')
    cnt = np.zeros((df_all.shape[0], 72), dtype=np.uint8)  # 26 English alphabets and 10 digits, for gene and variation
    str_list = list('abcdefghijklmnopqrstuvwxyz0123456789')

    cnt[:, :36] = np.asarray([[str(x).lower().count(c) for c in str_list] for x in df_all['Gene']])
    cnt[:, 36:] = np.asarray([[str(x).lower().count(c) for c in str_list] for x in df_all['Variation']])
    print('cnt.shape:', cnt.shape)

    print('Computing one hot feature(in character level) on gene/variation name...')
    # 26 English alphabets and 10 digits; max gene name: 9, max variation name: 55: for gene and variation
    onehot = np.zeros((df_all.shape[0], 112, 72), dtype=np.uint8)
    for idx, gene, var in zip(range(df_all.shape[0]), df_all['Gene'], df_all['Variation']):
        for i, c1 in zip(range(len(gene)), gene):
            if c1.lower() not in str_list:
                continue
            onehot[idx, i, str_list.index(c1.lower())] = 1
        for j, c2 in zip(range(len(var)), var):
            if c2.lower() not in str_list:
                continue
            onehot[idx, j + 56, str_list.index(c2.lower()) + 36] = 1

    onehot = onehot.reshape((df_all.shape[0], 112 * 72))
    print('onehot.shape:', onehot.shape)
    return cnt, onehot


def name_char1_8gram_svd(df_all):
    print('Computing gene/variation char level ngram(1, 8) + svd feature...')
    count_char = CountVectorizer(analyzer=u'char', ngram_range=(1, 8))
    count_svd = TruncatedSVD(n_components=20, n_iter=25, random_state=12)

    gene_count_feats = count_char.fit_transform(df_all['Gene'].apply(str))
    print('gene_count_feats.shape: ', gene_count_feats.shape)
    gene_svd_feats = count_svd.fit_transform(gene_count_feats)
    print('gene_svd_feats.shape: ', gene_svd_feats.shape)
    del gene_count_feats

    var_count_feats = count_char.fit_transform(df_all['Variation'].apply(str))
    print('var_count_feats.shape: ', var_count_feats.shape)
    var_svd_feats = count_svd.fit_transform(var_count_feats)
    print('var_svd_feats.shape: ', var_svd_feats.shape)
    del var_count_feats

    return gene_svd_feats, var_svd_feats


def get_entity_name_feats(train, test):
    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    cnt, onehot = name_count_onehot(df_all)

    train_feats, test_feats = np.vsplit(cnt, [train.shape[0], train.shape[0] + test.shape[0]])[:2]
    pd.DataFrame(data=train_feats).to_csv('data/features/char_count_train.csv', header=False, index=False)
    pd.DataFrame(data=test_feats).to_csv('data/features/char_count_test.csv', header=False, index=False)
    del train_feats, test_feats

    train_feats, test_feats = np.vsplit(onehot, [train.shape[0], train.shape[0] + test.shape[0]])[:2]
    pd.DataFrame(data=train_feats).to_csv('data/features/char_one_hot_train.csv', header=False, index=False)
    pd.DataFrame(data=test_feats).to_csv('data/features/char_one_hot_test.csv', header=False, index=False)
    del train_feats, test_feats

    gene_svd_feats, var_svd_feats = name_char1_8gram_svd(df_all)

    gene_svd_train, gene_svd_test = np.vsplit(gene_svd_feats, [train.shape[0], train.shape[0] + test.shape[0]])[:2]
    var_svd_train, var_svd_test = np.vsplit(var_svd_feats, [train.shape[0], train.shape[0] + test.shape[0]])[:2]

    gene_var_char1_8gram_svd_train = np.concatenate((gene_svd_train, var_svd_train), axis=1)
    print(gene_svd_train.shape, var_svd_train.shape, gene_var_char1_8gram_svd_train.shape)
    pd.DataFrame(data=gene_var_char1_8gram_svd_train).to_csv('data/features/gene_var_char1_8gram_svd_train.csv', header=False, index=False)

    gene_var_char1_8gram_svd_test = np.concatenate((gene_svd_test, var_svd_test), axis=1)
    print(gene_svd_test.shape, var_svd_test.shape, gene_var_char1_8gram_svd_test.shape)
    pd.DataFrame(data=gene_var_char1_8gram_svd_test).to_csv('data/features/gene_var_char1_8gram_svd_test.csv', header=False, index=False)


def _get_text(train_x):
    stopword_list = set(stopwords.words('english'))
    stopword_list.update(string.punctuation)

    docs = []
    for text in train_x:
        sentences = sent_tokenize(text)
        for sent in sentences:
            words = word_tokenize(sent.lower())
            token_list = [w for w in words if not w in stopword_list]
            docs.append(token_list)
    return docs


def train_word2vec(train_x, dim_):
    print('Training ' + str(dim_) + ' dim unigram word2vec model...')
    docs = _get_text(train_x)
    model = gensim.models.word2vec.Word2Vec(docs, size=dim_, alpha=0.05, window=30, min_count=5)
    model.save(open('data/models/word2vec/' + str(dim_) + '_dim/nips_w2v', 'wb'))


def word2vec_feats(train, test, dim):
    corpus = pd.concat((train, test), axis=0)
    train_word2vec(corpus['Text'].values, dim)
    model = gs.Word2Vec.load('data/models/word2vec/' + str(dim) + '_dim/nips_w2v')

    print('Computing word2vec feature...')
    train_w2v_mat = np.zeros((train.shape[0], dim * 2), dtype=np.float32)
    # gene/variation -> word2vec
    for idx, gene, var in zip(range(train.shape[0]), train['Gene'], train['Variation']):
        try:
            train_w2v_mat[idx, :dim] = model.wv[str(gene).lower()]
        except KeyError:
            pass
        try:
            train_w2v_mat[idx, dim:] = model.wv[str(var).lower()]
        except KeyError:
            pass
    pd.DataFrame(data=train_w2v_mat).to_csv('data/features/word2vec' + str(dim) + '_train.csv', header=False, index=False)

    test_w2v_mat = np.zeros((test.shape[0], dim * 2), dtype=np.float32)
    # gene/variation -> word2vec
    for idx, gene, var in zip(range(test.shape[0]), test['Gene'], test['Variation']):
        try:
            test_w2v_mat[idx, :dim] = model.wv[str(gene).lower()]
        except KeyError:
            pass
        try:
            test_w2v_mat[idx, dim:] = model.wv[str(var).lower()]
        except KeyError:
            pass
    pd.DataFrame(data=test_w2v_mat).to_csv('data/features/word2vec' + str(dim) + '_test.csv', header=False, index=False)


def _get_entity_occur_in_class(train):
    print('Building gene distribute dictionary...')
    gene_gps = train.groupby('Gene')
    gene_distribute_dict = dict()
    for gene, gp in gene_gps:
        gene_distribute_dict[gene.lower()] = dict(zip(list(range(1, 10)), list(np.zeros(9))))
        gene_distribute_dict[gene.lower()].update(dict(zip(list(gp['Class'].value_counts().index),
                                                           list(gp['Class'].value_counts().values / gp.shape[0]))))
    print('Saving gene distribute dictionary...')
    with open('data/intermediate/gene_distribute_dict.pkl', 'wb') as f:
        pickle.dump(gene_distribute_dict, f)

    print('Building variation distribute dictionary...')
    var_gps = train.groupby('Variation')
    var_distribute_dict = dict()
    for var, gp in var_gps:
        var_distribute_dict[var.lower()] = dict(zip(list(range(1, 10)), list(np.zeros(9))))
        var_distribute_dict[var.lower()].update(dict(zip(list(gp['Class'].value_counts().index),
                                                         list(gp['Class'].value_counts().values / gp.shape[0]))))
    print('Saving variation distribute dictionary...')
    with open('data/intermediate/var_distribute_dict.pkl', 'wb') as f:
        pickle.dump(var_distribute_dict, f)


def isint(value):
    try:
        int(value)
        return True

    except ValueError:
        return False


class Variation(object):
    def __init__(self):
        self.CH = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        self.NU = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}
        self.nu_lower = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}

        self.fs = 'fs'
        self.star = '*'
        self.deletion = 'Deletion'
        self.amplification = 'Amplification'
        self.overexpression = 'Overexpression'
        self.hypermethylation = 'Hypermethylation'
        self.wildtype = 'Wildtype'
        self.fusions = 'Fusions'
        self.splice = 'splice'
        self.del_ = 'del'
        self.ins = 'ins'
        self.delins = 'delins'
        self.dup = 'dup'
        self.es = 'es'
        self.trunc = 'trunc'

    def character_pattern(self, name):
        code = -1
        name = name.split(' ')
        if len(name) == 1:
            name = name[0]
            if name[0] in self.NU: # first character is an upper letter
                if name[-1] in self.NU: # last character is an upper letter
                    mid_name = name[1:-1]
                    if isint(mid_name): # int in the middle
                        code = 1
                        return code
                    elif self.delins in name: # 'delins' exits
                        if '_' in name:
                            code = 2
                            return code
                        else:
                            code = 3
                            return code
                    elif self.ins in name: # 'ins' exits
                        if '_' in name:
                            code = 4
                            return code
                        else:
                            code = 5
                            return code
                    elif self.star in name: # '*' exits
                        code = 6
                        return code
                elif name[-1] in self.nu_lower: # last character is an lower letter
                    if self.fs in name and name[-2:] == self.fs: # end with 'fs'
                        code = 7
                        return code
                    elif self.dup in name and name[-3:] == self.dup: # end with 'dup'
                        code = 8
                        return code
                    elif self.del_ in name and name[-3:] == self.del_: # end with 'del'
                        code = 9
                        return code
                    elif self.es in name and name[-2:] == self.es: # end with 'es'
                        code = 10
                        return code
                    elif self.ins in name and name[-3:] == self.ins: # end with 'ins'
                        code = 11
                        return code
                    elif self.splice in name and name[-6:] == self.splice: # end with 'splice'
                        code = 12
                        return code
                    elif self.trunc in name and name[-5:] == self.trunc: # end with 'trunc'
                        code = 13
                        return code
                    elif name == self.deletion:
                        code = 14
                        return code
                    elif name == self.amplification:
                        code = 15
                        return code
                    elif name == self.overexpression:
                        code = 16
                        return code
                    elif name == self.hypermethylation:
                        code = 17
                        return code
                    elif name == self.wildtype:
                        code = 18
                        return code
                    elif name == self.fusions:
                        code = 19
                        return code
                elif name[-1] == self.star: # end with '*'
                    if self.fs + self.star in name and name[-3:] == self.fs + self.star:
                        code = 20
                        return code
                    else:
                        code = 21
                        return code
                elif isint(name[-1]): # end with a number
                    if self.fs + self.star in name:
                        code = 22
                        return code
                    elif self.fs in name:
                        code = 23
                        return code
                    else:
                        code = 24
                        return code
            elif isint(name[0]): # first character is a number
                if self.splice in name and name[-6:] == self.splice:
                    code = 25
                    return code
                elif self.del_ in name and name[-3:] == self.del_:
                    code = 26
                    return code
                elif self.trunc in name and name[-5:] == self.trunc:
                    code = 27
                    return code
                elif self.ins in name:
                    code = 11
                    return code
            elif name[0] in self.nu_lower: # first character is a lower letter
                code = 28
                return code

        elif len(name) == 2: # two words name
            if name[-1] == 'Fusion': # 2 genes fusion
                code = 29
                return code
            elif name[0] == 'Truncating' and name[1] == 'Mutations':
                code = 30
                return code
            elif name[0] == 'Promoter':
                if name[1] == 'Mutations':
                    code = 31
                    return code
                elif name[1] == 'Hypermethylation':
                    code = 32
                    return code
            elif name[0] == 'Epigenetic' and name[1] == 'Silencing':
                code = 33
                return code
            elif 'Deletion' in name:
                code = 14
                return code

        elif len(name) > 2: # more than 2 words name
            if 'DNA' in name:
                code = 34
                return code
            elif 'Exon' in name:
                code = 35
                return code
            elif 'Truncating' in name and 'Mutations' in name:
                code = 36
                return code
            elif 'Splice' in name:
                code = 25
                return code
            elif 'Fusion' in name:
                code = 29
                return code
            elif 'Deletion' in name:
                code = 14
                return code
            else:
                code = 37
                return code
        if code == -1:
            code = 37
        return code


def _get_var_pattern(train, test_x):
    var = Variation()

    train['patn37'] = train['Variation'].map(lambda x:var.character_pattern(x))
    train.to_csv('data/intermediate/variation_pattern_train.csv', index=False)

    test_x['patn37'] = test_x['Variation'].map(lambda x: var.character_pattern(x))
    test_x.drop(['Text'], axis=1).to_csv('data/intermediate/variation_pattern_test.csv', index=False)


def _get_var_patn_distribute(train, test_x):
    _get_var_pattern(train, test_x)
    var_patn = pd.read_csv('data/intermediate/variation_pattern_train.csv')

    print('Building variation pattern37 distribute dictionary...')
    var_gps = var_patn.groupby('patn37')
    var_patn37_distribute_dict = dict()
    for var, gp in var_gps:
        var_patn37_distribute_dict[var] = dict(zip(list(range(1, 10)), list(np.zeros(9))))
        var_patn37_distribute_dict[var].update(dict(zip(list(gp['Class'].value_counts().index),
                                                                list(gp['Class'].value_counts().values / gp.shape[0]))))
    del var_gps

    print('Saving variation pattern37 distribute dictionary...')
    with open('data/intermediate/var_patn37_distribute_dict.pkl', 'wb') as f:
        pickle.dump(var_patn37_distribute_dict, f)


def _gene_var_rep_distribute(train, test_x, rep):
    print()
    print('*' * 50)
    _get_var_patn_distribute(train, test_x)

    print('Loading gene/variation distribute dictionary...')
    with open('data/intermediate/gene_distribute_dict.pkl', 'rb') as f:
        gene_distribute_dict = pickle.load(f)

    with open('data/intermediate/var_distribute_dict.pkl', 'rb') as f:
        var_distribute_dict = pickle.load(f)

    print('Loading variation pattern distribute dictionary...')
    with open('data/intermediate/var_patn37_distribute_dict.pkl', 'rb') as f:
        var_patn_distribute_dict = pickle.load(f)

    print('Computing gene/var distribution feature...')
    rep_train = pd.read_csv('data/features/' + rep + '_train.csv', header=None).loc[train['ID'].tolist()]
    dim = int(rep_train.shape[1] / 2)
    gene_train = np.zeros((train.shape[0], 9))
    var_train = np.zeros((train.shape[0], 9))

    for idx, row in train.iterrows():
        try:
            gene_train[idx, :] = np.array(list(gene_distribute_dict[row['Gene'].lower()].values()))
        except:
            pass

        code = Variation().character_pattern(row['Variation'])
        if code not in var_patn_distribute_dict:
            continue
        if code == 1:
            try:
                var_w2v = rep_train.loc[idx].values[dim:]
                dist = euclidean_distances(var_distribute_dict[row['Variation'].lower()], [var_w2v])[:, 0]
                var_train[idx, :] = dist / np.sum(dist)
            except:
                var_train[idx, :] = np.fromiter(iter(var_patn_distribute_dict[code].values()), dtype=float)
        else:
            var_train[idx, :] = np.fromiter(iter(var_patn_distribute_dict[code].values()), dtype=float)

    train_feat = np.concatenate((gene_train, var_train), axis=1)
    pd.DataFrame(data=train_feat).to_csv('data/features/gene_var_distribute_train.csv', header=None, index=False)

    rep_test = pd.read_csv('data/features/' + rep + '_test.csv', header=None).loc[train['ID'].tolist()]
    gene_test = np.zeros((test_x.shape[0], 9))
    var_test = np.zeros((test_x.shape[0], 9))

    for idx, row in test_x.iterrows():
        try:
            gene_test[idx, :] = np.array(list(gene_distribute_dict[row['Gene'].lower()].values()))
        except:
            pass

        code = Variation().character_pattern(row['Variation'])
        if code not in var_patn_distribute_dict:
            continue
        if code == 1:
            try:
                var_w2v = rep_test.loc[idx].values[dim:]
                dist = euclidean_distances(var_distribute_dict[row['Variation'].lower()], [var_w2v])[:, 0]
                var_test[idx, :] = dist / np.sum(dist)
            except:
                var_test[idx, :] = np.fromiter(iter(var_patn_distribute_dict[code].values()), dtype=float)
        else:
            var_test[idx, :] = np.fromiter(iter(var_patn_distribute_dict[code].values()), dtype=float)

    test_feat = np.concatenate((gene_test, var_test), axis=1)
    pd.DataFrame(data=test_feat).to_csv('data/features/gene_var_distribute_test.csv', header=None, index=False)


def get_gene_var_distribute(train_x, test_x, train_y):
    train = pd.concat((train_x.drop(['Text'], axis=1), pd.Series(data=train_y+1, name='Class')), axis=1)
    _get_entity_occur_in_class(train)

    if os.path.exists('data/features/word2vec100_train.csv') and \
            os.path.exists('data/features/word2vec100_test.csv'):
        _gene_var_rep_distribute(train, test_x, 'word2vec100')
    else:
        word2vec_feats(train_x, test_x, 100)
