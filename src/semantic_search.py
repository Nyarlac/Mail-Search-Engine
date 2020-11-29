import json
import time

import numpy as np
from nltk.stem import WordNetLemmatizer
from scipy.io import mmread

lemmatizer = WordNetLemmatizer()

# load mat
tfidf_sparse = mmread('../output/tf_idf.mtx')
tfidf = tfidf_sparse.toarray()
with open('../output/Doc_2_ID.json', 'r') as file:
    Doc_2_ID = json.load(file)
with open('../output/word_idx_1000.json', 'r') as file:
    word_idx = json.load(file)


def semantic_search():
    while (True):
        str_in = input('请输入查询词:')
        if str_in == '-1':
            break
        t0 = time.perf_counter()
        words = str_in.split(' ')
        input_arr = np.zeros([1000, 1])
        words_all = list(word_idx.keys())
        for word in words:
            word = lemmatize(word)
            input_arr[words_all.index(word)] += 1
        L = np.sqrt(input_arr.T.dot(input_arr))
        numer = np.sum(input_arr * tfidf, axis=0)
        denom = np.linalg.norm(tfidf, ord=None, axis=0)
        rank = (numer / (L * denom)).reshape(-1, )
        rank[np.isnan(rank)] = 0
        rank_list = list(rank)
        rank.sort()
        cnt = 0
        for score in rank[:rank.shape[0] - 11:-1]:
            if score == 0:
                break
            cnt += 1
            print("%d: %32s\tscore:%f" % (cnt, Doc_2_ID[str(rank_list.index(score))], score))
            rank_list[rank_list.index(score)] = 0
        print('花费时间：%f' % (time.perf_counter() - t0))


def lemmatize(word):
    word = lemmatizer.lemmatize(word, pos='n')
    word = lemmatizer.lemmatize(word, pos='v')
    return word


if __name__ == '__main__':
    semantic_search()
