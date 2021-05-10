import sys
import nltk
from nltk.corpus import stopwords
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import random
import re
from tqdm import tqdm

if len(sys.argv) < 2:
    sys.exit("Use: python remove_words.py <dataset>")

def clean_str(string):
    string = re.sub(r"[^\u0370-\u03ff\u1f00-\u1fff0-9a-zA-Z()#,_!?\'\`]", " ", string)
    string = re.sub(r",_\d", " ", string)
    string = re.sub(r"!_\d", " ", string)
    string = re.sub(r"__\d", " ", string)
    string = re.sub(r"«_\d", " ", string)
    string = re.sub(r"»_\d", " ", string)
    string = re.sub(r":_\d", " ", string)
    string = re.sub(r"=_\d", " ", string)
    string = re.sub(r" _\d", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

dataset = sys.argv[1]

try:
    least_freq = sys.argv[2]
except:
    least_freq = 5
    print('using default least word frequency = 5')


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words_greek = set(stopwords.words('greek'))
stop_words.update(stop_words_greek)
print(stop_words)


doc_content_list = []
with open('data/corpus/' + dataset + '.txt', 'rb') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('utf8'))

word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        if word.split('_')[0] not in stop_words and word_freq[word] >= least_freq:
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)


clean_corpus_str = '\n'.join(clean_docs)
with open('data/corpus/' + dataset + '.clean.txt', 'w', encoding='utf8') as f:
    f.write(clean_corpus_str)


len_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        if line == '\n':
            continue
        temp = line.strip().split()
        len_list.append(len(temp))

print('min_len : ' + str(min(len_list)))
print('max_len : ' + str(max(len_list)))
print('average_len : ' + str(sum(len_list)/len(len_list)))
