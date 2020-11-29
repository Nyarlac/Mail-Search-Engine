"""
    布尔检索
"""
import json
import time

#from nltk.stem import WordNetLemmatizer

#lemmatizer = WordNetLemmatizer()

# load data
with open('../output/Doc_2_ID.json', 'r') as file:
    Doc_2_ID = json.load(file)
with open('../output/word_idx_1000.json', 'r') as file:
    word_idx = json.load(file)


def bool_search():
    # loop
    idx_list = []
    while True:
        str_in = input("输入合取范式型查询: ")
        t0 = time.perf_counter()
        if str_in == 'exit':
            break
        str_in = str_in.split('and')
        if len(str_in) == 1:
            str_in = str_in[0]
            if 'not' not in str_in and 'or' not in str_in:
                str_in = lemmatize(str_in)
                idx_list = word_idx[str_in]
            else:
                items = str_in.split(' ')
                tmp = []
                is_not = is_or = 0
                for word in items:
                    if word == 'not':
                        is_not = 1
                    elif word == 'or':
                        is_or = 1
                    elif word.isalpha():
                        word = lemmatize(word)
                        if is_or == 1:
                            is_or = 0
                            if is_not == 0:
                                tmp = list_or(tmp, word_idx[word])
                            else:
                                tmp = list_or(tmp, list_not(word_idx[word]))
                                is_not = 0
                        else:
                            if is_not == 0:
                                tmp = word_idx[word]
                            else:
                                tmp = list_not(word_idx[word])
                                is_not = 0
                idx_list = tmp
        else:
            list_before_and = []
            for item in str_in:
                item = item.replace('(', '').replace(')', '').split(' ')
                is_or = is_not = 0
                tmp = []
                for word in item:
                    if word == 'not':
                        is_not = 1
                    elif word == 'or':
                        is_or = 1
                    elif word.isalpha():
                        word = lemmatize(word)
                        if is_or == 1:
                            is_or = 0
                            if is_not == 0:
                                tmp = list_or(tmp, word_idx[word])
                            else:
                                tmp = list_or(tmp, list_not(word_idx[word]))
                                is_not = 0
                        else:
                            if is_not == 0:
                                tmp = word_idx[word]
                            else:
                                tmp = list_not(word_idx[word])
                                is_not = 0
                list_before_and.append(tmp)
            for i in range(1, len(list_before_and)):
                list_before_and[0] = list_and(list_before_and[0], list_before_and[i])
            idx_list = list_before_and[0]
        for i in range(len(idx_list)):
            print(Doc_2_ID[str(idx_list[i])])
        print('共%d条记录，用时%fs' % (len(idx_list), time.perf_counter() - t0))


def list_and(a, b):
    len_a = len(a)
    len_b = len(b)
    result = []
    i = j = 0
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
    return result


def list_or(a, b):
    len_a = len(a)
    len_b = len(b)
    result = []
    i = j = 0
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        elif a[i] > b[j]:
            result.append(b[j])
            j += 1
    return result


def list_not(a):
    doc_num = int(list(Doc_2_ID.items())[-1][0])
    result = []
    i = j = 0
    while i <= doc_num and j < len(a):
        if a[j] == i:
            j += 1
            i += 1
        else:
            result.append(i)
            i += 1
    return result


def lemmatize(word):
    #word = lemmatizer.lemmatize(word, pos='n')
    #word = lemmatizer.lemmatize(word, pos='v')
    return word


if __name__ == '__main__':
    bool_search()
