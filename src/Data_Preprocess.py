import email
import json
import os
import time
import gensim

import numpy as np
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

lemmatizer = WordNetLemmatizer()


def preprocess_reverse_list():
    '''
        遍历文件，生成以下数据：
        - 文档路径&文档ID的词典Doc_2_ID
        - 单词&文档ID的词典word_idx
    '''
    Doc_2_ID = {}
    word_idx = {}
    ID_current = 0
    exs = 0
    _stopwords = stopwords.words('english')
    for root, dirs, files in os.walk('../dataset/maildir'):
        print(root)
        for file in files:
            file = os.path.join(root, file)
            with open(file, 'r') as file_obj:
                try:
                    msg = email.message_from_file(file_obj)
                    content = msg.get('subject') + '. '
                    for par in msg.walk():
                        if not par.is_multipart():
                            content += str(par.get_payload(decode=True), encoding='UTF-8')
                    content = content.lower()
                    Doc_2_ID[ID_current] = file
                    words = word_tokenize(content)
                    tags = pos_tag(words)
                    for word, tag in tags:
                        if word not in _stopwords:
                            pos = get_wordnet_pos(tag)
                            if pos is not None:
                                word = lemmatizer.lemmatize(word, pos=pos)
                                if word.isalpha():
                                    if word_idx.__contains__(word):
                                        if word_idx[word][-1] != ID_current:
                                            word_idx[word].append(ID_current)
                                    else:
                                        word_idx[word] = [ID_current]
                    ID_current += 1
                except:
                    exs += 1
    print(exs)
    with open('../output/Doc_2_ID.json', 'w') as file:
        json.dump(Doc_2_ID, file)
    with open('../output/word_idx.json', 'w') as file:
        json.dump(word_idx, file)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def keep_words_1000():
    with open('../output/word_idx.json', 'r') as file:
        word_idx = json.load(file)
    word_idx_1000 = {}
    word_cnt = {}
    cnt_list = []
    for k, v in word_idx.items():
        _len = len(v)
        word_cnt[k] = _len
        cnt_list.append(_len)
    cnt_list.sort(reverse=True)
    threshold = cnt_list[1000]
    for k, v in word_cnt.items():
        if v > threshold:
            word_idx_1000[k] = word_idx[k]
    with open('../output/word_idx_1000.json', 'w') as file:
        json.dump(word_idx_1000, file)


def preprocess_tf():
    with open('../output/Doc_2_ID.json', 'r') as file:
        Doc_2_ID = json.load(file)
    with open('../output/word_idx_1000.json', 'r') as file:
        word_idx = json.load(file)
    words_all = list(word_idx.keys())
    t0 = time.perf_counter()
    for idx, file in Doc_2_ID.items():
        if int(idx) >= 400000:
            if int(idx) % 1000 == 0:
                print(idx, 'time: %f s' % (time.perf_counter() - t0))
                if int(idx) != 400000 and int(idx) % 25000 == 0:
                    np.save('../output/tf_' + idx, mat)
            tmp = np.zeros([1000, 1])
            with open(file, 'r') as file_obj:
                msg = email.message_from_file(file_obj)
                content = msg.get('subject') + '. '
                for par in msg.walk():
                    if not par.is_multipart():
                        content += str(par.get_payload(decode=True), encoding='UTF-8')
                content = content.lower()
                words = word_tokenize(content)
                tags = pos_tag(words)
                for word, tag in tags:
                    pos = get_wordnet_pos(tag)
                    if pos is not None:
                        word = lemmatizer.lemmatize(word, pos=pos)
                        try:
                            tmp[words_all.index(word)] += 1
                        except:
                            pass
            if int(idx) % 25000 == 0:
                mat = tmp
            else:
                mat = np.append(mat, tmp, axis=1)
    np.save('../output/tf_' + idx, mat)


def preprocess_tfidf():
    for i in range(20):
        if i == 0:
            data = np.load('../output/tf_' + str((i + 1) * 25000) + '.npy')
        else:
            tmp = np.load('../output/tf_' + str((i + 1) * 25000) + '.npy')
            data = np.append(data, tmp, axis=1)
    tmp = np.load('../output/tf_517366.npy')
    data = np.append(data, tmp, axis=1)
    print('tf loaded!')
    with open('../output/word_idx_1000.json', 'r') as file:
        word_idx = json.load(file)
    df = []
    for _, idx in word_idx.items():
        df.append(len(idx))
    del word_idx
    df = np.array(df)
    idf = np.log10(517367 / df).reshape(-1, 1)
    tf = 1 + np.log10(data)
    tf[np.isinf(tf)] = 0
    tf_idf = idf * tf
    csr_mat = csr_matrix(tf_idf)
    mmwrite('../output/tf_idf', csr_mat)


def preprocess_word2vec():
    for i in range(20):
        if i == 0:
            data = np.load('../output/tf_' + str((i + 1) * 25000) + '.npy')
        else:
            tmp = np.load('../output/tf_' + str((i + 1) * 25000) + '.npy')
            data = np.append(data, tmp, axis=1)
    tmp = np.load('../output/tf_517366.npy')
    data = np.append(data, tmp, axis=1)
    print('tf loaded!')
    with open('../output/word_idx_1000.json', 'r') as file:
        word_idx = json.load(file)
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '../output/glove-twitter-25.txt')
    not_list_index = []
    word_vecs = []
    cnt = 0
    for k in word_idx.keys():
        try:
            tmp = model.wv[k]
            word_vecs.append(tmp.reshape([-1, 1]))
        except:
            word_vecs.append(np.ones([25, 1]))
            not_list_index.append(cnt)
        cnt += 1
    for i in range(data.shape[1]):
        if i % 10000 == 0:
            print(i)
        tmp = np.zeros([25, 1])
        for j in range(data.shape[0]):
            if j not in not_list_index:
                tmp += data[j, i] * word_vecs[j]
        if i == 0:
            word2vec_mat = tmp
        else:
            word2vec_mat = np.append(word2vec_mat, tmp, axis=1)
    np.save('../output/word2vec', word2vec_mat)


if __name__ == '__main__':
    # preprocess_reverse_list()
    # keep_words_1000()
    # preprocess_tf()
    # preprocess_tfidf()
    preprocess_word2vec()
