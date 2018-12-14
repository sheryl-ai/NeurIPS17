#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD, NMF
from collections import defaultdict
from math import log
import multiprocessing
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from collections import OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from itertools import chain
import jsonlines
from nltk.corpus import stopwords
import string
import re
import pickle
from sklearn import preprocessing
import gensim
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"


def read_corpus(docs, tokens_only=False):
    for i, line in enumerate(docs):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def train_doc2vec_dm(train_x, dim_, cores):
    print('Training ' + str(dim_) + ' dim unigram doc2vec model(PV-DM)...')
    train_corpus = list(read_corpus(train_x))

    model = gensim.models.doc2vec.Doc2Vec(size=dim_, window=10, min_count=1, hs=1, workers=cores, iter=55)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(open('data/models/doc2vec/' + str(dim_) + '_dim/nips_d2v_dm', 'wb'))


def train_doc2vec_dbow(train_x, dim_, cores):
    print('Training ' + str(dim_) + ' dim unigram doc2vec model(PV-DBOW)...')
    train_corpus = list(read_corpus(train_x))

    model = gensim.models.doc2vec.Doc2Vec(size=dim_, dm=0, window=10, min_count=1, hs=1, workers=cores, iter=55)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(open('data/models/doc2vec/' + str(dim_) + '_dim/nips_d2v_dbow', 'wb'))


def doc2vec_feats(train, test, dim1, dim2):
    train_corpus = train['Text'].values
    test_corpus = test['Text'].values
    corpus = pd.concat((train, test), axis=0)
    cores = multiprocessing.cpu_count()

    train_doc2vec_dm(corpus['Text'].values, dim1, cores)
    train_doc2vec_dbow(corpus['Text'].values, dim2, cores)

    print('Computing doc2vec(PV-DM + PV-DBOW) feature...')
    mode1_dm = gensim.models.doc2vec.Doc2Vec.load('data/models/doc2vec/' + str(dim1) + '_dim/nips_d2v_dm')
    mode1_dbow = gensim.models.doc2vec.Doc2Vec.load('data/models/doc2vec/' + str(dim2) + '_dim/nips_d2v_dbow')
    model = ConcatenatedDoc2Vec([mode1_dm, mode1_dbow])

    cor = list(read_corpus(train_corpus))
    train_feats = np.array(list(model.infer_vector(cor[idx].words) for idx in range(len(cor))))
    del cor
    pd.DataFrame(data=train_feats).to_csv('data/features/doc2vec_dm' + str(dim1) + '_dbow' + str(dim2) + '_train.csv', header=False, index=False)

    cor = list(read_corpus(test_corpus))
    test_feats = np.array(list(model.infer_vector(cor[idx].words) for idx in range(len(cor))))
    pd.DataFrame(data=test_feats).to_csv('data/features/doc2vec_dm' + str(dim1) + '_dbow' + str(dim2) + '_test.csv', header=False, index=False)


def doc_tfidf_svd_nmf(train, test):
    print('Applying document level tfidf + svd...')
    corpus = pd.concat((train, test), axis=0, ignore_index=True)['Text'].values
    tfidf_word_vector = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 3), stop_words='english')
    tfidf_svd = TruncatedSVD(n_components=50, n_iter=25, random_state=12)

    text_tfidf = tfidf_word_vector.fit_transform(corpus)
    print('text_tfidf.shape:', text_tfidf.shape)
    doc_tfidf_svd = tfidf_svd.fit_transform(text_tfidf)
    print('doc_tfidf_svd.shape:', doc_tfidf_svd.shape)

    train_feats, test_feats = np.vsplit(doc_tfidf_svd, [train.shape[0], train.shape[0] + test.shape[0]])[:2]
    pd.DataFrame(data=train_feats).to_csv('data/features/text_doc_svd_train.csv', header=False, index=False)
    pd.DataFrame(data=test_feats).to_csv('data/features/text_doc_svd_test.csv', header=False, index=False)
    del train_feats, test_feats

    print('Applying document level tfidf + nmf...')
    tfidf_nmf = NMF(n_components=60)
    doc_tfidf_nmf = tfidf_nmf.fit_transform(text_tfidf)
    print('doc_tfidf_nmf.shape:', doc_tfidf_nmf.shape)
    del text_tfidf

    train_feats, test_feats = np.vsplit(doc_tfidf_nmf, [train.shape[0], train.shape[0] + test.shape[0]])[:2]
    pd.DataFrame(data=train_feats).to_csv('data/features/text_doc_nmf_train.csv', header=False, index=False)
    pd.DataFrame(data=test_feats).to_csv('data/features/text_doc_nmf_test.csv', header=False, index=False)
    del train_feats, test_feats


def tf_lda(train, test):
    print('Extracting tf features for LDA...')
    count_vector = CountVectorizer(analyzer='word', stop_words='english')
    train_tf = count_vector.fit_transform(train)
    test_tf = count_vector.transform(test)
    print('train_tf.shape:', train_tf.shape)
    print('test_tf.shape:', test_tf.shape)

    print('Applying Latent Dirichlet Allocation...')
    lda_vector = LatentDirichletAllocation(n_components=50)
    train_lda = lda_vector.fit_transform(train_tf)
    test_lda = lda_vector.transform(test_tf)
    print('train_lda.shape:', train_lda.shape)
    print('test_lda.shape:', test_lda.shape)

    pd.DataFrame(data=train_lda).to_csv('data/features/tf_lda50_train.csv', header=False, index=False)
    pd.DataFrame(data=test_lda).to_csv('data/features/tf_lda50_test.csv', header=False, index=False)


def _merge_dict():
    verb_dict = [line.rstrip('\n') for line in open('data/intermediate/verb_base_list_wn.txt', 'r')]
    noun_dict = [line.rstrip('\n') for line in open('data/intermediate/noun_base_list_wn.txt', 'r')]
    adj_dict = [line.rstrip('\n') for line in open('data/intermediate/adjective_base_list_wn.txt', 'r')]
    adv_dict = [line.rstrip('\n') for line in open('data/intermediate/adverb_base_list_wn.txt', 'r')]

    dic = verb_dict + noun_dict + adj_dict + adv_dict
    with open('data/intermediate/unique_dict_all.txt', 'w') as f:
        for token in dic:
            f.write(token + '\n')

    cachedStopWords = stopwords.words("english")
    dic = [w for w in dic if w not in cachedStopWords]

    with open('data/intermediate/v_n_adj_adv_no_stopwords.txt', 'w') as f:
        for token in dic:
            f.write(token + '\n')


def _get_tf_dic(train_x, test_x):
    unique_dict_all = [line.rstrip('\n') for line in open('data/intermediate/v_n_adj_adv_no_stopwords.txt', 'r')]

    print('Generating TF dictionary...')
    vector = CountVectorizer(analyzer='word', stop_words='english')
    vector.fit_transform(train_x)
    train_dic = vector.vocabulary_.keys()

    vector.fit_transform(test_x)
    test_dic = vector.vocabulary_.keys()

    dic = set.intersection(set(unique_dict_all), set(train_dic).union(set(test_dic)))
    dic = sorted(dic)
    with open('data/intermediate/tf_unique_dict_all.txt', 'w') as f:
        for token in dic:
            f.write(token + '\n')

    return dic


def _word_occur_cls(dic):
    print('Merging dictionary...')
    df_intxn = pd.read_csv('data/intermediate/unique_dict_all_doc_rep_train.csv')[['Class'] + dic]

    gps = df_intxn.groupby('Class')
    cls_dic = []
    for cls, gp in gps:
        gp = gp.drop(['Class'], axis=1)
        s = gp.sum(axis=0)
        cls_dic.append(s.index[s > 0].tolist())
        cls_dic.append(s.index[s > 0].tolist())

    idf = defaultdict(list)
    dic_intxn = df_intxn.columns.tolist()
    for word in dic_intxn:
        for cls_idx in range(9):
            if word in cls_dic[cls_idx]:
                idf[word].append(cls_idx)

    idf_list = [(item[0], log(9 / len(item[1]))) for item in idf.items()]

    return idf, idf_list


def _get_tfidf(train_x, dic, idf_list):
    print('Computing TF-IDF feature...')
    vector = CountVectorizer(analyzer='word', stop_words='english', vocabulary=dict(zip(dic, range(len(dic)))))
    train_tf = vector.fit_transform(train_x).todense()
    tfidf = np.zeros_like(train_tf, dtype=float)

    for idx in range(len(idf_list)):
        tfidf[:, idx] = train_tf[:, idx] * idf_list[idx][1]

    return tfidf


def tf_custom_idf(train, test):
    _merge_dict()
    dic = _get_tf_dic(train, test)
    _, idf_list = _word_occur_cls(dic)
    train_feats = _get_tfidf(train, dic, idf_list)
    test_feats = _get_tfidf(test, dic, idf_list)
    pd.DataFrame(data=train_feats).to_csv('data/features/tf_custom_idf_train.csv', header=False, index=False)
    pd.DataFrame(data=test_feats).to_csv('data/features/tf_custom_idf_test.csv', header=False, index=False)


def text_win_7(train, test):
    print('Finding gene/variation appearance in corresponding text and extracting text window with size = 7...')
    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    word_vector = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1, 3), stop_words='english')
    char_Vector = TfidfVectorizer(strip_accents='unicode', analyzer='char', ngram_range=(1, 8), stop_words='english')

    svd_vector = TruncatedSVD(n_components=50, n_iter=25, random_state=12)

    text_win = np.zeros((df_all.shape[0], 2), dtype=object)

    for i, gene, var, text in zip(range(df_all.shape[0]), df_all['Gene'], df_all['Variation'], df_all['Text']):
        if gene in text:
            text_win[i, 0] = ' '.join(list(chain.from_iterable([text.split(' ')[idx - 3:idx + 4]
                                                                for idx, item in enumerate(text.split(' '))
                                                                if item == gene])))
        else:
            text_win[i, 0] = ''

        if var in text:
            text_win[i, 1] = ' '.join(list(chain.from_iterable([text.split(' ')[idx - 3:idx + 4]
                                                                for idx, item in enumerate(text.split(' '))
                                                                if item == var])))
        else:
            text_win[i, 1] = ''

    gene_word = word_vector.fit_transform(text_win[:, 0])
    print('gene_word.shape:', gene_word.shape)
    gene_word_svd = svd_vector.fit_transform(gene_word)
    print('gene_word_svd.shape:', gene_word_svd.shape)
    del gene_word

    gene_char = char_Vector.fit_transform(text_win[:, 0])
    print('gene_char.shape:', gene_char.shape)
    gene_char_svd = svd_vector.fit_transform(gene_char)
    print('gene_char_svd.shape:', gene_char_svd.shape)
    del gene_char

    var_word = word_vector.fit_transform(text_win[:, 1])
    print('var_word.shape:', var_word.shape)
    var_word_svd = svd_vector.fit_transform(var_word)
    print('var_word_svd.shape:', var_word_svd.shape)
    del var_word

    var_char = char_Vector.fit_transform(text_win[:, 1])
    print('var_char.shape:', var_char.shape)
    var_char_svd = svd_vector.fit_transform(var_char)
    print('var_char_svd.shape:', var_char_svd.shape)
    del var_char

    context_df_mat = np.concatenate((gene_word_svd, var_word_svd, gene_char_svd, var_char_svd), axis=1)

    col = [x + str(i) for x in ['Gene_word_', 'Var_word_', 'Gene_char_', 'Var_char_'] for i in range(gene_word_svd.shape[1])]
    context_df = pd.DataFrame(data=context_df_mat, columns=col)
    del gene_word_svd, var_word_svd, gene_char_svd, var_char_svd, context_df_mat

    train = context_df.iloc[:len(train)]
    test = context_df.iloc[len(train):]

    pd.DataFrame(data=train).to_csv('data/features/text_win_7_train.csv', header=False, index=False)
    pd.DataFrame(data=test).to_csv('data/features/text_win_7_test.csv', header=False, index=False)


def sent_tfidf_svd(train, test):
    print('Applying sentence level tfidf(word/char) + svd...')
    tfidf_word_vector = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 3), stop_words='english')
    tfidf_svd = TruncatedSVD(n_components=50, n_iter=25, random_state=12)

    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    sent_win = np.zeros((df_all.shape[0],), dtype=object)
    for i, gene, var, text in zip(range(df_all.shape[0]), df_all['Gene'], df_all['Variation'], df_all['Text']):
        sent_win[i] = ' '.join([sent for sent in sent_tokenize(text) if gene in sent or var in sent])

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

    word_svd_train, word_svd_test = np.vsplit(word_svd, [train.shape[0], train.shape[0] + test.shape[0]])[:2]
    char_svd_train, char_svd_test = np.vsplit(char_svd, [train.shape[0], train.shape[0] + test.shape[0]])[:2]

    sent_tfidf_word_char_svd_train = np.concatenate((word_svd_train, char_svd_train), axis=1)
    print(word_svd_train.shape, char_svd_train.shape, sent_tfidf_word_char_svd_train.shape)
    pd.DataFrame(sent_tfidf_word_char_svd_train).to_csv('data/features/sent_tfidf_word_char_svd_train.csv', header=False, index=False)

    sent_tfidf_word_char_svd_test = np.concatenate((word_svd_test, char_svd_test), axis=1)
    print(word_svd_test.shape, char_svd_test.shape, sent_tfidf_word_char_svd_test.shape)
    pd.DataFrame(data=sent_tfidf_word_char_svd_test).to_csv('data/features/sent_tfidf_word_char_svd_test.csv', header=False, index=False)


def get_text_mining_feats(train, test):
    doc_tfidf_svd_nmf(train, test)
    tf_lda(train['Text'].values, test['Text'].values)
    tf_custom_idf(train['Text'].values, test['Text'].values)
    text_win_7(train, test)
    sent_tfidf_svd(train, test)


def _pos_tagging_train(pos, stopword_list, train_x):
    variants_dict = OrderedDict()
    text_dict = OrderedDict()
    doc_pos = OrderedDict()
    token_base_list = set()
    token_to_id = dict()

    for _, row in train_x.iterrows():
        id = row['ID']
        variants_dict[int(id)] = tuple([row['Gene'], row['Variation']])
        doc_pos[int(id)] = dict()
        text_dict[int(id)] = row['Text']

    print('Getting token base form dictionary...')
    for i in range(len(variants_dict)):
        gene = variants_dict[i][0]
        variant = variants_dict[i][1]
        doc = text_dict[i]

        sentences = sent_tokenize(doc)
        for sent in sentences:
            if gene.lower() in sent.lower() or variant.lower() in sent.lower():
                # word tokenization
                word_list = word_tokenize(sent)
                # pos tagging
                word_list_with_pos_tag = pos_tag(word_list)

                for word_with_pos_tag in word_list_with_pos_tag:
                    token = word_with_pos_tag[0].lower()
                    token_tag = word_with_pos_tag[1]

                    if (pos[1] in token_tag) and (token not in stopword_list) and token.isalpha():
                        token_base = WordNetLemmatizer().lemmatize(token, pos[2])

                        # Check first POS in the wordnet
                        try:
                            first_pos = wn.synsets(token_base)[0].pos()
                            if first_pos == pos[2]:
                                token_base_list.add(token_base)

                                if token_base in doc_pos[i]:
                                    doc_pos[i][token_base] += 1
                                else:
                                    doc_pos[i][token_base] = 1
                        except:
                            pass

    token_base_list = sorted(token_base_list)
    print('Writing token base form dictionary...')
    with open('data/intermediate/' + pos[0] + '_base_list_wn.txt', 'w') as f:
        for token_base in token_base_list:
            f.write(token_base + '\n')

    index = 0
    for token_base in token_base_list:
        token_to_id[token_base] = index
        index += 1

    print('Computing token feature based on base form dictionary...')
    pos_matrix = np.zeros((len(variants_dict), len(token_base_list)))

    for key, value in doc_pos.items():
        for vk, vv in value.items():
            vk_id = token_to_id[vk]
            pos_matrix[key][vk_id] = vv

    pos_frame = pd.DataFrame(data=pos_matrix, columns=token_base_list)
    pos_frame.index.name = 'ID'
    pos_frame.insert(0, 'Variation', train_x['Variation'].tolist())
    pos_frame.insert(0, 'Gene', train_x['Gene'].tolist())
    pos_frame.to_csv('data/intermediate/' + pos[0] + '_doc_rep_train.csv')
    return token_base_list


def _pos_tagging_test(pos, stopword_list, test_x):
    variants_dict = OrderedDict()
    text_dict = OrderedDict()
    doc_pos = OrderedDict()
    pos_list = list()
    pos_to_id = dict()

    with open('data/intermediate/' + pos[0] + '_base_list_wn.txt', 'r') as f:
        index = 0
        for row in f:
            row = row.strip()
            pos_list.append(row)
            pos_to_id[row] = index
            index += 1

    for _, row in test_x.iterrows():
        id = row['ID']
        variants_dict[int(id)] = tuple([row['Gene'], row['Variation']])
        doc_pos[int(id)] = dict()
        text_dict[int(id)] = row['Text']

    for i in range(len(variants_dict)):
        gene = variants_dict[i][0]
        variant = variants_dict[i][1]
        doc = text_dict[i]

        sentences = sent_tokenize(doc)
        for sent in sentences:
            if gene.lower() in sent.lower() or variant.lower() in sent.lower():
                # word tokenization
                word_list = word_tokenize(sent)
                # pos tagging
                word_list_with_pos_tag = pos_tag(word_list)

                for word_with_pos_tag in word_list_with_pos_tag:
                    token = word_with_pos_tag[0].lower()
                    token_tag = word_with_pos_tag[1]

                    if (pos[1] in token_tag) and (token not in stopword_list) and token.isalpha():
                        token_base = WordNetLemmatizer().lemmatize(token, pos[2])

                        if token_base in pos_to_id:
                            if token_base in doc_pos[i]:
                                doc_pos[i][token_base] += 1
                            else:
                                doc_pos[i][token_base] = 1

    print('Computing token feature based on base form dictionary...')
    pos_matrix = np.zeros((len(variants_dict), len(pos_to_id)))

    for key, value in doc_pos.items():
        for vk, vv in value.items():
            vk_id = pos_to_id[vk]
            pos_matrix[key][vk_id] = vv

    pos_frame = pd.DataFrame(data=pos_matrix, columns=pos_list)
    pos_frame.index.name = 'ID'
    pos_frame.insert(0, 'Variation', test_x['Variation'].tolist())
    pos_frame.insert(0, 'Gene', test_x['Gene'].tolist())
    pos_frame.to_csv('data/intermediate/' + pos[0] + '_doc_rep_test.csv')


def pos_tagging_feats(train_x, test_x, train_y):
    stopword_list = set(stopwords.words('english'))
    stopword_list.update(string.punctuation)

    # verb(VB/v), noun(NN/n), adjective(JJ/a), adverb(RB/r)
    pos_list = [['verb', 'VB', 'v'], ['noun', 'NN', 'n'], ['adjective', 'JJ', 'a'], ['adverb', 'RB', 'r']]
    token_base_lists = []
    for pos in pos_list:
        print('Computing ' + pos[0] + ' feature...')
        token_base_lists += _pos_tagging_train(pos, stopword_list, train_x)
        _pos_tagging_test(pos, stopword_list, test_x)

    print('Computing unique dict all feature...')
    dic_all = sorted(set(token_base_lists))
    print('unique dict all:', len(dic_all))
    with open('data/intermediate/unique_dict_all.txt', 'w') as f:
        for token in dic_all:
            f.write(token + '\n')

    for data_type in ['train', 'test']:
        verb = pd.read_csv('data/intermediate/verb_doc_rep_' + data_type + '.csv')
        noun = pd.read_csv('data/intermediate/noun_doc_rep_' + data_type + '.csv')
        adj = pd.read_csv('data/intermediate/adjective_doc_rep_' + data_type + '.csv')
        adv = pd.read_csv('data/intermediate/adverb_doc_rep_' + data_type + '.csv')
        print('verb.shape:', verb.shape, ' noun.shape:', noun.shape, ' adjective.shape:', adj.shape, ' adverb.shape:', adv.shape)

        noun = noun.drop(['Gene', 'Variation'], axis=1)
        adj = adj.drop(['Gene', 'Variation'], axis=1)
        adv = adv.drop(['Gene', 'Variation'], axis=1)

        df = pd.merge(pd.merge(verb, noun, on='ID'), pd.merge(adj, adv, on='ID'), on='ID')

        if data_type == 'train':
            df_dic = pd.concat((df[['ID', 'Gene', 'Variation']], pd.Series(data=train_y), df[dic_all]), axis=1, ignore_index=True)
            df_dic.columns = ['ID', 'Gene', 'Variation', 'Class'] + dic_all
            df_dic.to_csv('data/intermediate/unique_dict_all_doc_rep_' + data_type + '.csv', index=False)
            df_dic.drop(['ID', 'Gene', 'Variation', 'Class'], axis=1).to_csv('data/features/unique_dict_all_doc_rep_' + data_type + '.csv', header=None, index=False)
        else:
            df_dic = pd.concat((df[['ID', 'Gene', 'Variation']], df[dic_all]), axis=1, ignore_index=True)
            df_dic.columns = ['ID', 'Gene', 'Variation'] + dic_all
            df_dic.to_csv('data/intermediate/unique_dict_all_doc_rep_' + data_type + '.csv', index=False)
            df_dic.drop(['ID', 'Gene', 'Variation'], axis=1).to_csv('data/features/unique_dict_all_doc_rep_' + data_type + '.csv', header=None, index=False)

        verb.drop(['ID', 'Gene', 'Variation'], axis=1).to_csv('data/features/verb_doc_rep_' + data_type + '.csv', header=None, index=False)
        noun.drop(['ID'], axis=1).to_csv('data/features/noun_doc_rep_' + data_type + '.csv', header=None, index=False)
        adj.drop(['ID'], axis=1).to_csv('data/features/adjective_doc_rep_' + data_type + '.csv', header=None, index=False)
        adv.drop(['ID'], axis=1).to_csv('data/features/adverb_doc_rep_' + data_type + '.csv', header=None, index=False)


def pos_tagging_nmf():
    print('Import gene/variation unique dict feature in associated document representation...')
    train_dict = pd.read_csv('data/features/unique_dict_all_doc_rep_train.csv', header=None).values
    test_dict = pd.read_csv('data/features/unique_dict_all_doc_rep_test.csv', header=None).values

    print('train_dict.shape:', train_dict.shape)
    print('test_dict.shape:', test_dict.shape)

    print('Applying Non-Negative Matrix Factorization(Frobenius)...')
    nmf_vector = NMF(n_components=60)
    pos_nmf_train = nmf_vector.fit_transform(train_dict)
    pos_nmf_test = nmf_vector.transform(test_dict)
    print('train_nmf_feats(Frobenius).shape:', pos_nmf_train.shape)
    print('test_nmf_feats(Frobenius).shape:', pos_nmf_test.shape)

    pd.DataFrame(data=pos_nmf_train).to_csv('data/features/unique_dict_all_nmf_train.csv', header=None, index=False)
    pd.DataFrame(data=pos_nmf_test).to_csv('data/features/unique_dict_all_nmf_test.csv', header=None, index=False)


def _bioentity_train(train_x):
    doc_disease_set = set()
    doc_chemical_set = set()
    doc_gene_set = set()
    doc_mutation_set = set()

    doc_bioentity_dic = OrderedDict([(i, {'chemical': {}, 'disease': {}, 'gene': {}, 'mutation': {}}) for i in range(train_x.shape[0])])

    print('Reading PubMed dictionaries...')
    pubmed_disease_dic = pd.read_table('data/pre_define/disease.tsv', header=None).set_index(0).to_dict()[1]
    pubmed_chemical_dic = pd.read_table('data/pre_define/chemical.tsv', header=None).set_index(0).to_dict()[1]
    pubmed_gene_dic = pd.read_table('data/pre_define/gene.tsv', header=None).set_index(0).to_dict()[1]
    pubmed_mutation_dic = pd.read_table('data/pre_define/mutation.tsv', header=None).set_index(0).to_dict()[1]

    print('Counting bioentity occurrence id...')
    for i, row in train_x.iterrows():
        doc = row['Text'].lower()

        for disease_name, disease_id in pubmed_disease_dic.items():
            if disease_name in doc:
                # To count distinct disease ids appear in the corpora
                doc_disease_set.add(disease_id)
                n = doc.count(disease_name)
                if disease_id in doc_bioentity_dic[i]['disease']:
                    doc_bioentity_dic[i]['disease'][disease_id] = doc_bioentity_dic[i]['disease'][disease_id] + n
                else:
                    doc_bioentity_dic[i]['disease'][disease_id] = n

        for chemical_name, chemical_id in pubmed_chemical_dic.items():
            if chemical_name in doc:
                # To count distinct chemical ids appear in the corpora
                doc_chemical_set.add(chemical_id)
                n = doc.count(chemical_name)
                if chemical_id in doc_bioentity_dic[i]['chemical']:
                    doc_bioentity_dic[i]['chemical'][chemical_id] = doc_bioentity_dic[i]['chemical'][chemical_id] + n
                else:
                    doc_bioentity_dic[i]['chemical'][chemical_id] = n

        for gene_name, gene_id in pubmed_gene_dic.items():
            if gene_name in doc:
                # To count distinct gene ids appear in the corpora
                doc_gene_set.add(gene_id)
                n = doc.count(gene_name)
                if gene_id in doc_bioentity_dic[i]['gene']:
                    doc_bioentity_dic[i]['gene'][gene_id] = doc_bioentity_dic[i]['gene'][gene_id] + n
                else:
                    doc_bioentity_dic[i]['gene'][gene_id] = n

        for mutation_name, mutation_id in pubmed_mutation_dic.items():
            if mutation_name in doc:
                # To count distinct disease ids appear in the corpora
                doc_mutation_set.add(mutation_id)
                n = doc.count(mutation_name)
                if mutation_id in doc_bioentity_dic[i]['mutation']:
                    doc_bioentity_dic[i]['mutation'][mutation_id] = doc_bioentity_dic[i]['mutation'][mutation_id] + n
                else:
                    doc_bioentity_dic[i]['mutation'][mutation_id] = n

    doc_disease_name_id_map = dict(zip(doc_disease_set, range(len(doc_disease_set))))
    doc_chemical_name_id_map = dict(zip(doc_chemical_set, range(len(doc_chemical_set))))
    doc_gene_name_id_map = dict(zip(doc_gene_set, range(len(doc_gene_set))))
    doc_mutation_name_id_map = dict(zip(doc_mutation_set, range(len(doc_mutation_set))))

    print('Writing bioentity name-id mapping...')
    with open('data/intermediate/disease_name_id_map.tsv', 'w') as f:
        for name, index in doc_disease_name_id_map.items():
            f.write(str(index) + '\t' + name + '\n')

    with open('data/intermediate/chemical_name_id_map.tsv', 'w') as f:
        for name, index in doc_chemical_name_id_map.items():
            f.write(str(index) + '\t' + name + '\n')

    with open('data/intermediate/gene_name_id_map.tsv', 'w') as f:
        for name, index in doc_gene_name_id_map.items():
            f.write(str(index) + '\t' + name + '\n')

    with open('data/intermediate/mutation_name_id_map.tsv', 'w') as f:
        for name, index in doc_mutation_name_id_map.items():
            f.write(str(index) + '\t' + name + '\n')

    print('Counting bioentity occurrence matrix...')
    disease_matrix = np.zeros((train_x.shape[0], len(doc_disease_set)))
    chemical_matrix = np.zeros((train_x.shape[0], len(doc_chemical_set)))
    gene_matrix = np.zeros((train_x.shape[0], len(doc_gene_set)))
    mutation_matrix = np.zeros((train_x.shape[0], len(doc_mutation_set)))

    for doc_id, bioentities in doc_bioentity_dic.items():
        disease_list = bioentities['disease']
        chemical_list = bioentities['chemical']
        gene_list = bioentities['gene']
        mutation_list = bioentities['mutation']

        for disease_id, value in disease_list.items():
            disease_index = doc_disease_name_id_map[disease_id]
            disease_matrix[doc_id][disease_index] = value

        for chemical_id, value in chemical_list.items():
            chemical_index = doc_chemical_name_id_map[chemical_id]
            chemical_matrix[doc_id][chemical_index] = value

        for gene_id, value in gene_list.items():
            gene_index = doc_gene_name_id_map[gene_id]
            gene_matrix[doc_id][gene_index] = value

        for mutation_id, value in mutation_list.items():
            mutation_index = doc_mutation_name_id_map[mutation_id]
            mutation_matrix[doc_id][mutation_index] = value

    disease_frame = pd.DataFrame(data=disease_matrix, columns=list(doc_disease_set))
    disease_frame.index.name = 'ID'
    disease_frame.insert(0, 'Variation', train_x['Variation'].tolist())
    disease_frame.insert(0, 'Gene', train_x['Gene'].tolist())
    disease_frame.to_csv('data/intermediate/disease_doc_rep_train.csv')
    pd.DataFrame(data=disease_matrix).to_csv('data/features/disease_doc_rep_train.csv', header=None, index=False)

    chemical_frame = pd.DataFrame(data=chemical_matrix, columns=list(doc_chemical_set))
    chemical_frame.index.name = 'ID'
    chemical_frame.insert(0, 'Variation', train_x['Variation'].tolist())
    chemical_frame.insert(0, 'Gene', train_x['Gene'].tolist())
    chemical_frame.to_csv('data/intermediate/chemical_doc_rep_train.csv')
    pd.DataFrame(data=chemical_matrix).to_csv('data/features/chemical_doc_rep_train.csv', header=None, index=False)

    gene_frame = pd.DataFrame(data=gene_matrix, columns=list(doc_gene_set))
    gene_frame.index.name = 'ID'
    gene_frame.insert(0, 'Variation', train_x['Variation'].tolist())
    gene_frame.insert(0, 'Gene', train_x['Gene'].tolist())
    gene_frame.to_csv('data/intermediate/gene_doc_rep_train.csv')
    pd.DataFrame(data=gene_matrix).to_csv('data/features/gene_doc_rep_train.csv', header=None, index=False)

    mutation_frame = pd.DataFrame(data=mutation_matrix, columns=list(doc_mutation_set))
    mutation_frame.index.name = 'ID'
    mutation_frame.insert(0, 'Variation', train_x['Variation'].tolist())
    mutation_frame.insert(0, 'Gene', train_x['Gene'].tolist())
    mutation_frame.to_csv('data/intermediate/mutation_doc_rep_train.csv')
    pd.DataFrame(data=mutation_matrix).to_csv('data/features/mutation_doc_rep_train.csv', header=None, index=False)


def _bioentity_test(test_x):
    print('Reading PubMed dictionaries...')
    pubmed_disease_dic = pd.read_table('data/pre_define/disease.tsv', header=None).set_index(0).to_dict()[1]
    pubmed_chemical_dic = pd.read_table('data/pre_define/chemical.tsv', header=None).set_index(0).to_dict()[1]
    pubmed_gene_dic = pd.read_table('data/pre_define/gene.tsv', header=None).set_index(0).to_dict()[1]
    pubmed_mutation_dic = pd.read_table('data/pre_define/mutation.tsv', header=None).set_index(0).to_dict()[1]

    print('Reading bioentity name-id mapping...')
    disease_to_id = {val: key for key, val in pd.read_table('data/intermediate/disease_name_id_map.tsv', header=None).set_index(0).to_dict()[1].items()}
    chemical_to_id = {val: key for key, val in pd.read_table('data/intermediate/chemical_name_id_map.tsv', header=None).set_index(0).to_dict()[1].items()}
    gene_to_id = {val: key for key, val in pd.read_table('data/intermediate/gene_name_id_map.tsv', header=None).set_index(0).to_dict()[1].items()}
    mutation_to_id = {val: key for key, val in pd.read_table('data/intermediate/mutation_name_id_map.tsv', header=None).set_index(0).to_dict()[1].items()}

    doc_disease_all = OrderedDict()
    doc_chemical_all = OrderedDict()
    doc_gene_all = OrderedDict()
    doc_mutation_all = OrderedDict()

    print('Counting bioentity occurrence id...')
    for i, row in test_x.iterrows():
        doc = row['Text'].lower()

        doc_disease_dic = dict()
        doc_chemical_dic = dict()
        doc_gene_dic = dict()
        doc_mutation_dic = dict()

        for disease_name, disease_id in pubmed_disease_dic.items():
            if disease_name in doc and disease_id in disease_to_id:
                n = doc.count(disease_name)
                if disease_id in doc_disease_dic:
                    doc_disease_dic[disease_id] = doc_disease_dic[disease_id] + n
                else:
                    doc_disease_dic[disease_id] = n

        for chemical_name, chemical_id in pubmed_chemical_dic.items():
            if chemical_name in doc and chemical_id in chemical_to_id:
                n = doc.count(chemical_name)
                if chemical_id in doc_chemical_dic:
                    doc_chemical_dic[chemical_id] = doc_chemical_dic[chemical_id] + n
                else:
                    doc_chemical_dic[chemical_id] = n

        for gene_name, gene_id in pubmed_gene_dic.items():
            if gene_name in doc and gene_id in gene_to_id:
                n = doc.count(gene_name)
                if gene_id in doc_gene_dic:
                    doc_gene_dic[gene_id] = doc_gene_dic[gene_id] + n
                else:
                    doc_gene_dic[gene_id] = n

        for mutation_name, mutation_id in pubmed_mutation_dic.items():
            if mutation_name in doc and mutation_id in mutation_to_id:
                n = doc.count(mutation_name)
                if mutation_id in doc_mutation_dic:
                    doc_mutation_dic[mutation_id] = doc_mutation_dic[mutation_id] + n
                else:
                    doc_mutation_dic[mutation_id] = n

    disease_matrix = np.zeros((test_x.shape[0], len(disease_to_id)))
    chemical_matrix = np.zeros((test_x.shape[0], len(chemical_to_id)))
    gene_matrix = np.zeros((test_x.shape[0], len(gene_to_id)))
    mutation_matrix = np.zeros((test_x.shape[0], len(mutation_to_id)))

    for doc_id, values in doc_disease_all.items():
        for disease_id, count in values.items():
            disease_index = disease_to_id[disease_id]
            disease_matrix[doc_id][disease_index] = count

    for doc_id, values in doc_chemical_all.items():
        for chemical_id, count in values.items():
            chemical_index = chemical_to_id[chemical_id]
            chemical_matrix[doc_id][chemical_index] = count

    for doc_id, values in doc_gene_all.items():
        for gene_id, count in values.items():
            gene_index = gene_to_id[gene_id]
            gene_matrix[doc_id][gene_index] = count

    for doc_id, values in doc_mutation_all.items():
        for mutation_id, count in values.items():
            mutation_index = mutation_to_id[mutation_id]
            mutation_matrix[doc_id][mutation_index] = count

    disease_frame = pd.DataFrame(data=disease_matrix, columns=list(disease_to_id.keys()))
    disease_frame.index.name = 'ID'
    disease_frame.insert(0, 'Variation', test_x['Variation'].tolist())
    disease_frame.insert(0, 'Gene', test_x['Gene'].tolist())
    disease_frame.to_csv('data/intermediate/disease_doc_rep_test.csv')
    pd.DataFrame(data=disease_matrix).to_csv('data/features/disease_doc_rep_test.csv', header=None, index=False)

    chemical_frame = pd.DataFrame(data=chemical_matrix, columns=list(chemical_to_id.keys()))
    chemical_frame.index.name = 'ID'
    chemical_frame.insert(0, 'Variation', test_x['Variation'].tolist())
    chemical_frame.insert(0, 'Gene', test_x['Gene'].tolist())
    chemical_frame.to_csv('data/intermediate/chemical_doc_rep_test.csv')
    pd.DataFrame(data=chemical_matrix).to_csv('data/features/chemical_doc_rep_test.csv', header=None, index=False)

    gene_frame = pd.DataFrame(data=gene_matrix, columns=list(gene_to_id.keys()))
    gene_frame.index.name = 'ID'
    gene_frame.insert(0, 'Variation', test_x['Variation'].tolist())
    gene_frame.insert(0, 'Gene', test_x['Gene'].tolist())
    gene_frame.to_csv('data/intermediate/gene_doc_rep_test.csv')
    pd.DataFrame(data=gene_matrix).to_csv('data/features/gene_doc_rep_test.csv', header=None, index=False)

    mutation_frame = pd.DataFrame(data=mutation_matrix, columns=list(mutation_to_id.keys()))
    mutation_frame.index.name = 'ID'
    mutation_frame.insert(0, 'Variation', test_x['Variation'].tolist())
    mutation_frame.insert(0, 'Gene', test_x['Gene'].tolist())
    mutation_frame.to_csv('data/intermediate/mutation_doc_rep_test.csv')
    pd.DataFrame(data=mutation_matrix).to_csv('data/features/mutation_doc_rep_test.csv', header=None, index=False)


def bioentity_feats(train_x, test_x):
    _bioentity_train(train_x)
    _bioentity_test(test_x)


def _build_dict():
    oncokb = pd.read_table('data/pre_define/Actionable.txt')
    print('Building drug dictionary...')
    drug = set(chain(*[val.split(', ') for val in set(chain(*[val.lower().split(' + ') for val in oncokb['Drugs(s)'].tolist()]))]))
    print('Saving drug dictionary...')
    with open('data/intermediate/drug.dict.pkl', 'wb') as f:
        pickle.dump(drug, f)

    print('Building tumor dictionary...')
    tumor = set(chain(*[val.lower().split(', ') for val in set(oncokb['Cancer Type'].tolist())]))
    print('Saving tumor dictionary...')
    with open('data/intermediate/tumor.dict.pkl', 'wb') as f:
        pickle.dump(tumor, f)

    print('Building title dictionary...')
    stopword_list = set(stopwords.words('english'))
    stopword_list.update(string.punctuation)
    stopword_list.update([line.rstrip('\n') for line in open('data/pre_define/pubmed_stopword_list.txt', 'r')])

    title = []
    with jsonlines.open('data/pre_define/pubmed.jsonl') as reader:
        for obj in reader:
            words = word_tokenize(obj['MedlineCitation']['Article']['ArticleTitle'])
            try:
                title += [re.match('^[a-zA-Z0-9]+(-[a-zA-Z0-9]+)*$', w.lower()).group(0) for w in words if w not in stopword_list]
            except AttributeError:
                pass

    with open('data/pre_define/nips.10k.dict.pkl', 'rb') as f:
        nips10k = pickle.load(f)

    title = set([w for w in title if not w.isdigit() and len(w) != 1 and w in nips10k])

    print('Saving title dictionary...')
    with open('data/intermediate/title.dict.pkl', 'wb') as f:
        pickle.dump(title, f)
    print('drug.dict.shape:', len(drug), 'tumor.dict.shape:', len(tumor), 'title.dict.shape:', len(title))


def get_dict_feats(dic_name, train, test):
    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    print('Computing ' + dic_name + ' dictionary feature...')
    with open('data/intermediate/' + dic_name + '.dict.pkl', 'rb') as f:
        dic = pickle.load(f)

    for word in dic:
        df_all[word] = df_all['Text'].map(lambda x: str(x).count(word))

    df_all = df_all.drop(['ID', 'Gene', 'Variation', 'Text'], axis=1)
    train = df_all.iloc[:len(train)]
    test = df_all.iloc[len(train):]
    train.to_csv('data/features/' + dic_name + '_train.csv', header=None, index=False)
    test.to_csv('data/features/' + dic_name + '_test.csv', header=None, index=False)


def pubmed_feats(train, test):
    _build_dict()

    for dic_name in ['drug', 'tumor', 'title']:
        get_dict_feats(dic_name, train, test)


def gene_var_text_len(train, test):
    print('Counting gene/variation appearance in corresponding text...')
    df_all = pd.concat((train, test), axis=0, ignore_index=True)

    print('Calculating gene/variation string length at specific length threshold...')
    for i in range(56):
        df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
        df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')

    print('Computing gene/variation name length(char/word level)...')
    df_all['Gene_len'] = df_all['Gene'].map(lambda x: len(str(x)))
    df_all['Gene_words'] = df_all['Gene'].map(lambda x: len(str(x).split(' ')))

    df_all['Variation_len'] = df_all['Variation'].map(lambda x: len(str(x)))
    df_all['Variation_words'] = df_all['Variation'].map(lambda x: len(str(x).split(' ')))

    print('Computing text length(char/word level)...')
    df_all['Text_len'] = df_all['Text'].map(lambda x: len(str(x)))
    df_all['Text_words'] = df_all['Text'].map(lambda x: len(str(x).split(' ')))

    for c in df_all.columns:
        if df_all[c].dtype == 'object':
            if c in ['Gene', 'Variation']:
                lbl = preprocessing.LabelEncoder()
                df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)
                df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
                df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
            elif c != 'Text':
                lbl = preprocessing.LabelEncoder()
                df_all[c] = lbl.fit_transform(df_all[c].values)

    df_feat = df_all.drop(['ID', 'Gene', 'Variation', 'Text'], axis=1)
    train_feat = df_feat.iloc[:len(train)]
    test_feat = df_feat.iloc[len(train):]
    print('train.shape: ', train_feat.shape, ' test.shape: ', test_feat.shape)

    train_feat.to_csv('data/features/gene_var_text_word_str_len_train.csv', header=False, index=False)
    test_feat.to_csv('data/features/gene_var_text_word_str_len_test.csv', header=False, index=False)


def text_id(train, test):
    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    df_all['Text_len'] = df_all['Text'].map(lambda x: len(str(x)))
    lbl = preprocessing.LabelEncoder()
    df_all['Text_id'] = lbl.fit_transform(df_all['Text_len'].values)

    df_feat = df_all.drop(['ID', 'Gene', 'Variation', 'Text', 'Text_len'], axis=1)
    train_feat = df_feat.iloc[:len(train)]
    test_feat = df_feat.iloc[len(train):]
    print('train.shape: ', train_feat.shape, ' test.shape: ', test_feat.shape)

    train_feat.to_csv('data/features/text_id_train.csv', header=False, index=False)
    test_feat.to_csv('data/features/text_id_test.csv', header=False, index=False)
