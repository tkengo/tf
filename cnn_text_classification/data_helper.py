import numpy
import itertools
import os.path
import sys
from collections import Counter
from constants import *
from datetime import datetime

def split_word(tagger, content):
    word = tagger.parse(content).split(' ')
    word = [ w.strip().decode('utf-8') for w in word ]
    return word

def one_hot_vec(index):
    v = numpy.zeros(NUM_CLASSES)
    v[index] = 1
    return v

def padding(contents, max_word_count):
    padded_contents = []
    for i in xrange(len(contents)):
        content = contents[i]
        padded_contents.append(content + [ '<PAD/>' ] * (max_word_count - len(content)))

    return padded_contents

def load_data_and_labels_and_dictionaries():
    if os.path.exists(DATA_FILE) and os.path.exists(LABEL_FILE) and os.path.exists(DICTIONARY_FILE):
        data         = numpy.load(DATA_FILE)
        labels       = numpy.load(LABEL_FILE)
        dictionaries = numpy.load(DICTIONARY_FILE)

    else:
        import MeCab

        lines = [ l.split("\t") for l in list(open(RAW_FILE).readlines()) ]
        t = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

        contents = [ split_word(t, l[1]) for l in lines ]
        contents = padding(contents, max([ len(c) for c in contents ]))
        labels   = [ one_hot_vec(int(l[0]) - 1) for l in lines ]

        ctr = Counter(itertools.chain(*contents))
        dictionaries     = [ c[0] for c in ctr.most_common() ]
        dictionaries_inv = { c: i for i, c in enumerate(dictionaries) }

        data = [ [ dictionaries_inv[word] for word in content ] for content in contents ]

        data         = numpy.array(data)
        labels       = numpy.array(labels)
        dictionaries = numpy.array(dictionaries)

        numpy.save(DATA_FILE,       data)
        numpy.save(LABEL_FILE,      labels)
        numpy.save(DICTIONARY_FILE, dictionaries)

    return [ data, labels, dictionaries ]

def log(content):
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print time + ': ' + content
    sys.stdout.flush()
