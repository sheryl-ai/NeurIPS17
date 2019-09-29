#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD, NMF
from itertools import chain
import numpy as np
import pandas as pd
import ahocorasick  # https://pypi.python.org/pypi/pyahocorasick/#api-overview

from feature import document_mining


def _get_entity_text(train, flag='train'):
    if flag == 'train':
        print('*' * 20 + 'neurips 3321 + stage1_test 368' + '*' * 20)
    else:
        print('*' * 20 + 'stage 2 986' + '*' * 20)
    gene_list = set(train['Gene'].tolist())
    var_list = set(train['Variation'].tolist())
    print('unique gene in ' + flag + ':', len(gene_list),
          ' unique variation in ' + flag + ':', len(var_list))

    sentences = set(list(chain.from_iterable(train['Text'].map(lambda x: sent_tokenize(x)))))
    print('number of unique sentences in all samples:', len(sentences))

    print('Building gene tree...')
    gene_tree = ahocorasick.Automaton()
    for idx, gene in enumerate(gene_list):
        gene_tree.add_word(gene.lower(), (idx, gene.lower()))

    gene_tree.make_automaton()

    print('Assinging gene occurrence sentences to gene tree...')
    gene_dict = dict()
    for sent in sentences:
        word_list = word_tokenize(sent.lower())
        for word in word_list:
            if word in gene_tree:
                if word in gene_dict:
                    gene_dict[word] += sent
                else:
                    gene_dict[word] = sent

    print('Writing gene occurrence sentences to file...')
    df_gene = pd.DataFrame.from_dict(gene_dict, orient='index').reset_index()
    df_gene.columns = ['Gene', 'Text']
    df_gene.to_csv('data/intermediate/' + flag + '_gene_text', index=False, sep='|')

    print('Building variation tree...')
    var_tree = ahocorasick.Automaton()
    for idx, var in enumerate(var_list):
        var_tree.add_word(var.lower(), (idx, var.lower()))

    var_tree.make_automaton()

    print('Assinging variation occurrence sentences to variation tree...')
    var_dict = dict()
    for sent in sentences:
        word_list = word_tokenize(sent.lower())
        for word in word_list:
            if word in var_tree:
                if word in var_dict:
                    var_dict[word] += sent
                else:
                    var_dict[word] = sent

    print('Writing variation occurrence sentences to file...')
    df_gene = pd.DataFrame.from_dict(var_dict, orient='index').reset_index()
    df_gene.columns = ['Variation', 'Text']
    df_gene.to_csv('data/intermediate/' + flag + '_variation_text', index=False, sep='|')


def _get_tf_dic(train_x, test_x, flag='gene'):
    unique_dict_all = [line.rstrip('\n') for line in open('data/intermediate/v_n_adj_adv_no_stopwords.txt', 'r')]

    print('Generating TF dictionary...')
    vector = CountVectorizer(analyzer='word', stop_words='english')
    vector.fit_transform(train_x)
    train_dic = vector.vocabulary_.keys()

    vector.fit_transform(test_x)
    test_dic = vector.vocabulary_.keys()

    dic = set.intersection(set(unique_dict_all), set(train_dic).union(set(test_dic)))
    dic = sorted(dic)
    with open('data/intermediate/' + flag + '_tf_unique_dict_all.txt', 'w') as f:
        for token in dic:
            f.write(token + '\n')
    return dic


def _convert(train, test):
    gv_text_doc_svd = pd.read_csv('data/intermediate/gv_text_doc_svd.csv', header=None)
    gv_text_doc_nmf = pd.read_csv('data/intermediate/gv_text_doc_nmf.csv', header=None)
    gv_sent_tfidf_word_char_svd = pd.read_csv('data/intermediate/gv_sent_tfidf_word_char_svd.csv', header=None)
    gv_gene_tf_lda50 = pd.read_csv('data/intermediate/gv_gene_tf_lda50.csv', header=None)
    gv_var_tf_lda50 = pd.read_csv('data/intermediate/gv_var_tf_lda50.csv', header=None)
    gv_gene_tf_custom_idf = pd.read_csv('data/intermediate/gv_gene_tf_custom_idf.csv', header=None)
    gv_var_tf_custom_idf = pd.read_csv('data/intermediate/gv_var_tf_custom_idf.csv', header=None)
    gv_gene_unique_tf_custom_idf = pd.read_csv('data/intermediate/gv_gene_unique_tf_custom_idf.csv', header=None)
    gv_var_unique_tf_custom_idf = pd.read_csv('data/intermediate/gv_var_unique_tf_custom_idf.csv', header=None)
    gv_gene_var_intxn_tf_custom_idf = pd.read_csv('data/intermediate/gv_gene_var_intxn_tf_custom_idf.csv', header=None)

    gv_text_doc_svd_train = np.zeros([train.shape[0], 2 * (gv_text_doc_svd.shape[1] - 1)], dtype='float64')
    gv_text_doc_nmf_train = np.zeros([train.shape[0], 2 * (gv_text_doc_nmf.shape[1] - 1)], dtype='float64')
    gv_sent_tfidf_word_char_svd_train = np.zeros([train.shape[0], 2 * (gv_sent_tfidf_word_char_svd.shape[1] - 1)], dtype='float64')
    gv_tf_lda50_train = np.zeros([train.shape[0], (gv_gene_tf_lda50.shape[1] + gv_var_tf_lda50.shape[1] - 2)], dtype='float64')
    gv_tf_custom_idf_train = np.zeros([train.shape[0], (gv_gene_tf_custom_idf.shape[1] + gv_var_tf_custom_idf.shape[1] - 2)], dtype='float64')
    gv_tf_custom_idf_unique_train = np.zeros([train.shape[0], (gv_gene_unique_tf_custom_idf.shape[1] + gv_var_unique_tf_custom_idf.shape[1] + gv_gene_var_intxn_tf_custom_idf.shape[1] - 3)], dtype='float64')

    for idx, row in train.iterrows():
        gene = row['Gene'].lower()
        variation = row['Variation'].lower()
        try:
            gv_text_doc_svd_train[idx, :(gv_text_doc_svd.shape[1] - 1)] = gv_text_doc_svd[gv_text_doc_svd[0] == gene].values[0][1:].astype(float)
            gv_text_doc_nmf_train[idx, :(gv_text_doc_nmf.shape[1] - 1)] = gv_text_doc_nmf[gv_text_doc_nmf[0] == gene].values[0][1:].astype(float)
            gv_sent_tfidf_word_char_svd_train[idx, :(gv_sent_tfidf_word_char_svd.shape[1] - 1)] = gv_sent_tfidf_word_char_svd[gv_sent_tfidf_word_char_svd[0] == gene].values[0][1:].astype(float)
            gv_tf_lda50_train[idx, :(gv_gene_tf_lda50.shape[1] - 1)] = gv_gene_tf_lda50[gv_gene_tf_lda50[0] == gene].values[0][1:].astype(float)
            gv_tf_custom_idf_train[idx, :(gv_gene_tf_custom_idf.shape[1] - 1)] = gv_gene_tf_custom_idf[gv_gene_tf_custom_idf[0] == gene].values[0][1:].astype(float)
        except:
            pass
        try:
            gv_text_doc_svd_train[idx, (gv_text_doc_svd.shape[1] - 1):] = gv_text_doc_svd[gv_text_doc_svd[0] == variation].values[0][1:].astype(float)
            gv_text_doc_nmf_train[idx, (gv_text_doc_nmf.shape[1] - 1):] = gv_text_doc_nmf[gv_text_doc_nmf[0] == variation].values[0][1:].astype(float)
            gv_sent_tfidf_word_char_svd_train[idx, (gv_sent_tfidf_word_char_svd.shape[1] - 1):] = gv_sent_tfidf_word_char_svd[gv_sent_tfidf_word_char_svd[0] == variation].values[0][1:].astype(float)
            gv_tf_lda50_train[idx, (gv_gene_tf_lda50.shape[1] - 1):] = gv_var_tf_lda50[gv_var_tf_lda50[0] == variation].values[0][1:].astype(float)
            gv_tf_custom_idf_train[idx, (gv_gene_tf_custom_idf.shape[1] - 1):] = gv_var_tf_custom_idf[gv_var_tf_custom_idf[0] == variation].values[0][1:].astype(float)
        except:
            pass

        if gene in gv_gene_unique_tf_custom_idf[0].tolist() or variation in gv_var_unique_tf_custom_idf[0].tolist():
            try:
                gv_tf_custom_idf_unique_train[idx, :(gv_gene_unique_tf_custom_idf.shape[1] - 1)] = gv_gene_unique_tf_custom_idf[gv_gene_unique_tf_custom_idf[0] == gene].values[0][1:].astype(float)
                gv_tf_custom_idf_unique_train[idx, gv_gene_unique_tf_custom_idf.shape[1]: (gv_gene_unique_tf_custom_idf.shape[1] + gv_var_unique_tf_custom_idf.shape[1])] = gv_var_unique_tf_custom_idf[gv_var_unique_tf_custom_idf[0] == variation].values[0][1:].astype(float)
            except:
                pass
        else:
            try:
                gv_tf_custom_idf_unique_train[idx, -gv_gene_var_intxn_tf_custom_idf.shape[1]:] = gv_gene_var_intxn_tf_custom_idf[gv_gene_var_intxn_tf_custom_idf[0] == gene].values[0][1:].astype(float)
            except:
                try:
                    gv_tf_custom_idf_unique_train[idx, -gv_gene_var_intxn_tf_custom_idf.shape[1]:] = gv_gene_var_intxn_tf_custom_idf[gv_gene_var_intxn_tf_custom_idf[0] == variation].values[0][1:].astype(float)
                except:
                    pass

    pd.DataFrame(data=gv_text_doc_svd_train).to_csv('data/features/gv_text_doc_svd_train.csv', header=None, index=False)
    pd.DataFrame(data=gv_text_doc_nmf_train).to_csv('data/features/gv_text_doc_nmf_train.csv', header=None, index=False)
    pd.DataFrame(data=gv_sent_tfidf_word_char_svd_train).to_csv('data/features/gv_sent_tfidf_word_char_svd_train.csv', header=None, index=False)
    pd.DataFrame(data=gv_tf_lda50_train).to_csv('data/features/gv_tf_lda50_train.csv', header=None, index=False)
    pd.DataFrame(data=gv_tf_custom_idf_train).to_csv('data/features/gv_tf_custom_idf_train.csv', header=None, index=False)
    pd.DataFrame(data=gv_tf_custom_idf_unique_train).to_csv('data/features/gv_tf_custom_idf_unique_train.csv', header=None, index=False)

    gv_text_doc_svd_test = np.zeros([test.shape[0], 2 * (gv_text_doc_svd.shape[1] - 1)], dtype='float64')
    gv_text_doc_nmf_test = np.zeros([test.shape[0], 2 * (gv_text_doc_nmf.shape[1] - 1)], dtype='float64')
    gv_sent_tfidf_word_char_svd_test = np.zeros([test.shape[0], 2 * (gv_sent_tfidf_word_char_svd.shape[1] - 1)], dtype='float64')
    gv_tf_lda50_test = np.zeros([test.shape[0], (gv_gene_tf_lda50.shape[1] + gv_var_tf_lda50.shape[1] - 2)], dtype='float64')
    gv_tf_custom_idf_test = np.zeros([test.shape[0], (gv_gene_tf_custom_idf.shape[1] + gv_var_tf_custom_idf.shape[1] - 2)], dtype='float64')
    gv_tf_custom_idf_unique_test = np.zeros([test.shape[0], (gv_gene_unique_tf_custom_idf.shape[1] + gv_var_unique_tf_custom_idf.shape[1] + gv_gene_var_intxn_tf_custom_idf.shape[1] - 3)], dtype='float64')

    for idx, row in test.iterrows():
        gene = row['Gene'].lower()
        variation = row['Variation'].lower()
        try:
            gv_text_doc_svd_test[idx, :(gv_text_doc_svd.shape[1] - 1)] = gv_text_doc_svd[gv_text_doc_svd[0] == gene].values[0][1:].astype(float)
            gv_text_doc_nmf_test[idx, :(gv_text_doc_nmf.shape[1] - 1)] = gv_text_doc_nmf[gv_text_doc_nmf[0] == gene].values[0][1:].astype(float)
            gv_sent_tfidf_word_char_svd_test[idx, :(gv_sent_tfidf_word_char_svd.shape[1] - 1)] = gv_sent_tfidf_word_char_svd[gv_sent_tfidf_word_char_svd[0] == gene].values[0][1:].astype(float)
            gv_tf_lda50_test[idx, :(gv_gene_tf_lda50.shape[1] - 1)] = gv_gene_tf_lda50[gv_gene_tf_lda50[0] == gene].values[0][1:].astype(float)
            gv_tf_custom_idf_test[idx, :(gv_gene_tf_custom_idf.shape[1] - 1)] = gv_gene_tf_custom_idf[gv_gene_tf_custom_idf[0] == gene].values[0][1:].astype(float)
        except:
            pass
        try:
            gv_text_doc_svd_test[idx, (gv_text_doc_svd.shape[1] - 1):] = gv_text_doc_svd[gv_text_doc_svd[0] == variation].values[0][1:].astype(float)
            gv_text_doc_nmf_test[idx, (gv_text_doc_nmf.shape[1] - 1):] = gv_text_doc_nmf[gv_text_doc_nmf[0] == variation].values[0][1:].astype(float)
            gv_sent_tfidf_word_char_svd_test[idx, (gv_sent_tfidf_word_char_svd.shape[1] - 1):] = gv_sent_tfidf_word_char_svd[gv_sent_tfidf_word_char_svd[0] == variation].values[0][1:].astype(float)
            gv_tf_lda50_test[idx, (gv_gene_tf_lda50.shape[1] - 1):] = gv_var_tf_lda50[gv_var_tf_lda50[0] == variation].values[0][1:].astype(float)
            gv_tf_custom_idf_test[idx, (gv_gene_tf_custom_idf.shape[1] - 1):] = gv_var_tf_custom_idf[gv_var_tf_custom_idf[0] == variation].values[0][1:].astype(float)
        except:
            pass

        if gene in gv_gene_unique_tf_custom_idf[0].tolist() or variation in gv_var_unique_tf_custom_idf[0].tolist():
            try:
                gv_tf_custom_idf_unique_test[idx, :(gv_gene_unique_tf_custom_idf.shape[1] - 1)] = gv_gene_unique_tf_custom_idf[gv_gene_unique_tf_custom_idf[0] == gene].values[0][1:].astype(float)
                gv_tf_custom_idf_unique_test[idx, gv_gene_unique_tf_custom_idf.shape[1]: (gv_gene_unique_tf_custom_idf.shape[1] + gv_var_unique_tf_custom_idf.shape[1])] = gv_var_unique_tf_custom_idf[gv_var_unique_tf_custom_idf[0] == variation].values[0][1:].astype(float)
            except:
                pass
        else:
            try:
                gv_tf_custom_idf_unique_test[idx, -gv_gene_var_intxn_tf_custom_idf.shape[1]:] = gv_gene_var_intxn_tf_custom_idf[gv_gene_var_intxn_tf_custom_idf[0] == gene].values[0][1:].astype(float)
            except:
                try:
                    gv_tf_custom_idf_unique_test[idx, -gv_gene_var_intxn_tf_custom_idf.shape[1]:] = gv_gene_var_intxn_tf_custom_idf[gv_gene_var_intxn_tf_custom_idf[0] == variation].values[0][1:].astype(float)
                except:
                    pass

    pd.DataFrame(data=gv_text_doc_svd_test).to_csv('data/features/gv_text_doc_svd_test.csv', header=None, index=False)
    pd.DataFrame(data=gv_text_doc_nmf_test).to_csv('data/features/gv_text_doc_nmf_test.csv', header=None, index=False)
    pd.DataFrame(data=gv_sent_tfidf_word_char_svd_test).to_csv('data/features/gv_sent_tfidf_word_char_svd_test.csv', header=None, index=False)
    pd.DataFrame(data=gv_tf_lda50_test).to_csv('data/features/gv_tf_lda50_test.csv', header=None, index=False)
    pd.DataFrame(data=gv_tf_custom_idf_test).to_csv('data/features/gv_tf_custom_idf_test.csv', header=None, index=False)
    pd.DataFrame(data=gv_tf_custom_idf_unique_test).to_csv('data/features/gv_tf_custom_idf_unique_test.csv', header=None, index=False)


def gene_var_share(train, test):
    print('Counting gene/variation appearance in corresponding text...')
    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]),
                                        axis=1)
    df_all['Variation_Share'] = df_all.apply(
        lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

    print('Sorting unique gene/variation and removing longer variation(word length >=2)...')
    gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
    gen_var_lst = [x for x in gen_var_lst if len(x.split(' ')) == 1]

    print('Encoding labels and calculating gene/variation/text string/word length...')
    for gen_var_lst_itm in gen_var_lst:
        df_all['GV_' + str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))

    df_feat = df_all.drop(['ID', 'Gene', 'Variation', 'Text'], axis=1)
    df_feat.iloc[:len(train)].to_csv('data/features/gene_var_share_train.csv', header=False, index=False)
    df_feat.iloc[len(train):].to_csv('data/features/gene_var_share_test.csv', header=False, index=False)


def gene_var_text_relation():
    print('Loading gene/variation text...')
    train_gene_text = pd.read_csv('data/intermediate/train_gene_text', sep='|')
    train_var_text = pd.read_csv('data/intermediate/train_variation_text', sep='|')

    test_gene_text = pd.read_csv('data/intermediate/test_gene_text', sep='|')
    test_var_text = pd.read_csv('data/intermediate/test_variation_text', sep='|')
    print('train_gene_text.shape:', train_gene_text.shape, 'train_var_text.shape:', train_var_text.shape,
          'test_gene_text.shape:', test_gene_text.shape, 'test_var_text.shape:', test_var_text.shape)

    gene_text = pd.concat((train_gene_text, test_gene_text), axis=0, ignore_index=True)
    gene_text.columns = ['Entity', 'Text']
    gps = gene_text.groupby('Entity')
    entity = []
    text = []
    for val, gp in gps:
        if gp.shape[0] != 1:
            entity.append(gp['Entity'].values[0])
            text.append(gp['Text'].sum())
    for e in entity:
        gene_text = gene_text[gene_text['Entity'] != e]
    gene_text = gene_text.append(pd.DataFrame({'Entity': entity, 'Text': text}), ignore_index=True)
    del entity, text, gps

    var_text = pd.concat((train_var_text, test_var_text), axis=0, ignore_index=True)
    var_text.columns = ['Entity', 'Text']
    gps = var_text.groupby('Entity')
    entity = []
    text = []
    for val, gp in gps:
        if gp.shape[0] != 1:
            entity.append(gp['Entity'].values[0])
            text.append(gp['Text'].sum())
    for e in entity:
        var_text = var_text[var_text['Entity'] != e]
    var_text = var_text.append(pd.DataFrame({'Entity': entity, 'Text': text}), ignore_index=True)
    del entity, text, gps

    gene_var_text = pd.concat((gene_text, var_text), axis=0, ignore_index=True)
    print(gene_text.shape, var_text.shape, gene_var_text.shape)

    print('Applying document level tfidf + svd...')
    tfidf_word_vector = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 3), stop_words='english')
    tfidf_svd = TruncatedSVD(n_components=50, n_iter=25, random_state=12)

    text_tfidf = tfidf_word_vector.fit_transform(gene_var_text['Text'].values)
    print('text_tfidf:', text_tfidf.shape)
    text_doc_svd = tfidf_svd.fit_transform(text_tfidf)
    print('text_doc_svd:', text_doc_svd.shape)
    pd.concat((gene_var_text['Entity'], pd.DataFrame(data=text_doc_svd)), axis=1, ignore_index=True).to_csv('data/intermediate/gv_text_doc_svd.csv', header=False, index=False)

    print('Applying document level tfidf + nmf...')
    tfidf_nmf = NMF(n_components=60)
    text_doc_nmf = tfidf_nmf.fit_transform(text_tfidf)
    print('text_doc_nmf:', text_doc_nmf.shape)
    pd.concat((gene_var_text['Entity'], pd.DataFrame(data=text_doc_nmf)), axis=1, ignore_index=True).to_csv('data/intermediate/gv_text_doc_nmf.csv', header=False, index=False)
    del text_tfidf

    print('Applying sentence level tfidf(word/char) + svd...')
    sent_win = np.zeros((gene_var_text.shape[0],), dtype=object)
    for i, text in enumerate(gene_var_text['Text'].tolist()):
        sent_win[i] = ' '.join([sent for sent in sent_tokenize(text)])

    tfidf_char_vector = TfidfVectorizer(strip_accents='unicode', analyzer='char', ngram_range=(1, 8), stop_words='english')

    word = tfidf_word_vector.fit_transform(sent_win)
    print('word.shape:', word.shape)
    word_svd = tfidf_svd.fit_transform(word)
    print('word_svd.shape:', word_svd.shape)
    del word

    char = tfidf_char_vector.fit_transform(sent_win)
    print('char.shape:', char.shape)
    del sent_win
    char_svd = tfidf_svd.fit_transform(char)
    print('char_svd.shape:', char_svd.shape)
    del char

    sent_tfidf_word_char_svd = np.concatenate((word_svd, char_svd), axis=1)
    del word_svd, char_svd
    print('sent_tfidf_word_char_svd:', sent_tfidf_word_char_svd.shape)
    pd.concat((gene_var_text['Entity'], pd.DataFrame(data=sent_tfidf_word_char_svd)), axis=1, ignore_index=True).to_csv('data/intermediate/gv_sent_tfidf_word_char_svd.csv', header=False, index=False)

    print('Extracting tf features on gene text for LDA...')
    count_vector = CountVectorizer(analyzer='word', stop_words='english')
    gene_train_tf = count_vector.fit_transform(train_gene_text['Text'].values)
    gene_test_tf = count_vector.transform(test_gene_text['Text'].values)
    print('gene_train_tf:', gene_train_tf.shape)
    print('gene_test_tf:', gene_test_tf.shape)

    print('Applying Latent Dirichlet Allocation on gene text...')
    lda_vector = LatentDirichletAllocation(n_components=50)
    gene_train_lda = lda_vector.fit_transform(gene_train_tf)
    gene_test_lda = lda_vector.transform(gene_test_tf)
    print('gene_train_lda:', gene_train_lda.shape)
    print('gene_test_lda:', gene_test_lda.shape)
    del gene_train_tf, gene_test_tf
    gene_lda = np.concatenate((gene_train_lda, gene_test_lda), axis=0)
    del gene_train_lda, gene_test_lda
    gene_lda_df = pd.concat((pd.concat([train_gene_text['Gene'], test_gene_text['Gene']], axis=0, ignore_index=True),
                             pd.DataFrame(data=gene_lda)), axis=1, ignore_index=True)

    # merge same entities
    gps = gene_lda_df.groupby(0)
    entity = []
    vec = []
    for val, gp in gps:
        if gp.shape[0] != 1:
            entity.append(gp[0].values[0])
            vec.append(sum(gp.values[:, 1:].astype(float)))
    for e in entity:
        gene_lda_df = gene_lda_df[gene_lda_df[0] != e]
    gene_lda_df = gene_lda_df.append(pd.concat([pd.DataFrame(data=entity), pd.DataFrame(data=vec)], axis=1, ignore_index=True))
    del entity, vec, gps

    gene_lda_df.to_csv('data/intermediate/gv_gene_tf_lda50.csv', header=False, index=False)
    del gene_lda

    print('Extracting tf features on variation text for LDA...')
    var_train_tf_feats = count_vector.fit_transform(train_var_text['Text'].values)
    var_test_tf_feats = count_vector.transform(test_var_text['Text'].values)
    print('var_train_tf_feats:', var_train_tf_feats.shape)
    print('var_test_tf_feats:', var_test_tf_feats.shape)

    print('Applying Latent Dirichlet Allocation on variation text...')
    var_train_lda_feats = lda_vector.fit_transform(var_train_tf_feats)
    var_test_lda_feats = lda_vector.transform(var_test_tf_feats)
    print('var_train_lda_feats:', var_train_lda_feats.shape)
    print('var_test_lda_feats:', var_test_lda_feats.shape)
    del var_train_tf_feats, var_test_tf_feats
    var_lda = np.concatenate((var_train_lda_feats, var_test_lda_feats), axis=0)
    del var_train_lda_feats, var_test_lda_feats
    var_lda_df = pd.concat((pd.concat([train_var_text['Variation'], test_var_text['Variation']], axis=0, ignore_index=True),
                            pd.DataFrame(data=var_lda)), axis=1, ignore_index=True)
    # merge same entities
    gps = var_lda_df.groupby(0)
    entity = []
    vec = []
    for val, gp in gps:
        if gp.shape[0] != 1:
            entity.append(gp[0].values[0])
            vec.append(sum(gp.values[:, 1:].astype(float)))
    for e in entity:
        var_lda_df = var_lda_df[var_lda_df[0] != e]
    var_lda_df = var_lda_df.append(pd.concat([pd.DataFrame(data=entity), pd.DataFrame(data=vec)], axis=1, ignore_index=True))
    del entity, vec, gps
    var_lda_df.to_csv('data/intermediate/gv_var_tf_lda50.csv', header=False, index=False)
    del var_lda

    print('Applying TF custom idf feature on gene text...')
    gene_dic = _get_tf_dic(train_gene_text['Text'].values, test_gene_text['Text'].values, flag='gene')
    _, gene_idf_list = document_mining._word_occur_cls(gene_dic)
    gene_tfidf = document_mining._get_tfidf(gene_text['Text'].values, gene_dic, gene_idf_list)
    pd.concat((gene_text['Entity'], pd.DataFrame(data=gene_tfidf)), axis=1, ignore_index=True).to_csv('data/intermediate/gv_gene_tf_custom_idf.csv', header=False, index=False)

    print('Applying TF custom idf feature on variation text...')
    var_dic = _get_tf_dic(train_var_text['Text'].values, test_var_text['Text'].values, flag='variation')
    _, var_idf_list = document_mining._word_occur_cls(var_dic)
    var_tfidf = document_mining._get_tfidf(var_text['Text'].values, var_dic, var_idf_list)
    pd.concat((var_text['Entity'], pd.DataFrame(data=var_tfidf)), axis=1, ignore_index=True).to_csv('data/intermediate/gv_var_tf_custom_idf.csv', header=False, index=False)
    del gene_dic, var_dic, gene_idf_list, var_idf_list, gene_tfidf, var_tfidf

    print('Applying TF custom idf feature on built gene/var dictionary...')
    gene_dic = set([line.rstrip('\n') for line in open('data/intermediate/gene_tf_unique_dict_all.txt', 'r')])
    var_dic = set([line.rstrip('\n') for line in open('data/intermediate/variation_tf_unique_dict_all.txt', 'r')])
    gene_var_dic_intxn = set(gene_dic).intersection(set(var_dic))

    gene_unique_dic = list(gene_dic - gene_var_dic_intxn)
    _, gene_idf_list = document_mining._word_occur_cls(gene_unique_dic)
    gene_tfidf = document_mining._get_tfidf(gene_text['Text'].values, gene_unique_dic, gene_idf_list)
    pd.concat((gene_text['Entity'], pd.DataFrame(data=gene_tfidf)), axis=1, ignore_index=True).to_csv('data/intermediate/gv_gene_unique_tf_custom_idf.csv', header=False, index=False)

    var_unique_dic = list(var_dic - gene_var_dic_intxn)
    _, var_idf_list = document_mining._word_occur_cls(var_unique_dic)
    var_tfidf = document_mining._get_tfidf(var_text['Text'].values, var_unique_dic, var_idf_list)
    pd.concat((var_text['Entity'], pd.DataFrame(data=var_tfidf)), axis=1, ignore_index=True).to_csv('data/intermediate/gv_var_unique_tf_custom_idf.csv', header=False, index=False)

    _, idf_list = document_mining._word_occur_cls(list(gene_var_dic_intxn))
    tfidf = document_mining._get_tfidf(gene_var_text['Text'].values, list(gene_var_dic_intxn), idf_list)
    pd.concat((gene_var_text['Entity'], pd.DataFrame(data=tfidf)), axis=1, ignore_index=True).to_csv('data/intermediate/gv_gene_var_intxn_tf_custom_idf.csv', header=False, index=False)


def get_relation_mining_feats(train, test):
    gene_var_share(train, test)

    _get_entity_text(train, flag='train')
    _get_entity_text(test, flag='test')

    gene_var_text_relation()
    _convert(train, test)
