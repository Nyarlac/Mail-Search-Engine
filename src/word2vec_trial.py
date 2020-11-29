import gensim
import json
import time
import numpy as np
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# load data
model = gensim.models.KeyedVectors.load_word2vec_format(
        '../output/glove-twitter-25.txt')
word2vec_mat = np.load('../output/word2vec.npy')
with open('../output/Doc_2_ID.json', 'r') as file:
    Doc_2_ID = json.load(file)
with open('../output/word_idx_1000.json', 'r') as file:
    word_idx = json.load(file)

def semantic_search_word2vec():
    while (True):
        str_in = input('请输入查询词:')
        if str_in == '-1':
            break
        t0 = time.perf_counter()
        words = str_in.split(' ')
        input_arr = np.zeros([25, 1])
        for word in words:
            word = lemmatize(word)
            input_arr += model.wv[word].reshape(-1,1)
        L = np.sqrt(input_arr.T.dot(input_arr))
        numer = np.sum(input_arr * word2vec_mat, axis=0)
        denom = np.linalg.norm(word2vec_mat, ord=None, axis=0)
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
    semantic_search_word2vec()
