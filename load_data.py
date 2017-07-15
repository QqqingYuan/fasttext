__author__ = 'PC-LiNing'

import codecs
from gensim import corpora
from collections import  defaultdict
import numpy


def invert_dict(d):
    return dict([(v,k) for k,v in d.items()])


def stats_words_labels():
    words = []
    labels = []
    f = codecs.open('cooking.train',encoding='utf-8')
    for line in f.readlines():
        line = line.strip('\n').strip()
        tokens = line.split()
        current_labels = []
        current_words = []
        for token in tokens:
            if token.startswith("__label__"):
                current_labels.append(token)
            else:
                if token != "1":
                    current_words.append(token)
        words.append(current_words)
        labels.append(current_labels)
    f.close()
    f = codecs.open('cooking.valid',encoding='utf-8')
    for line in f.readlines():
        line = line.strip('\n').strip()
        tokens = line.split()
        current_labels = []
        current_words = []
        for token in tokens:
            if token.startswith("__label__"):
                current_labels.append(token)
            else:
                if token != "1":
                    current_words.append(token)
        words.append(current_words)
        labels.append(current_labels)

    dic_words = corpora.Dictionary(words)
    max_words = max([len(text) for text in words])
    min_words = min([len(text) for text in words])
    print("words dict: "+str(len(dic_words)))
    print(str(max_words)+" "+str(min_words))
    dic_words.save('words.dict')
    dic_labels = corpora.Dictionary(labels)
    max_labels = max([len(text) for text in labels])
    min_labels = min([len(text) for text in labels])
    print("labels dict: "+str(len(dic_labels)))
    print(str(max_labels)+" "+str(min_labels))
    dic_labels.save('labels.dict')


def transfer_label(labels):
    label_size = 736
    vec = numpy.zeros(label_size)
    for ids in labels:
        vec[ids] = 1.0
    return vec


def get_data(file):
    words_dict = invert_dict(corpora.Dictionary.load('words.dict'))
    labels_dict = invert_dict(corpora.Dictionary.load('labels.dict'))
    paths_length = numpy.load('paths_length.npy')
    train_data = []
    train_label = []
    train_label_num = []
    labels = []
    train_paths_length = []
    f = codecs.open(file,encoding='utf-8')
    for line in f.readlines():
        line = line.strip('\n').strip()
        tokens = line.split()
        current_labels = []
        current_words = []
        for token in tokens:
            if token.startswith("__label__"):
                current_labels.append(labels_dict[token])
            else:
                if token != "1":
                    current_words.append(words_dict[token])
        current_path_length = [paths_length[label] for label in current_labels]
        train_paths_length.append(current_path_length)
        train_label_num.append(len(current_labels))
        labels.extend(current_labels)
        train_data.append(current_words)
        train_label.append(current_labels)
    # get counts
    # counts = [labels.count(item) for item in set(labels)]
    return train_data,train_label,labels,train_label_num,train_paths_length


def load_cooking_data():
    train_data, train_label,train_counts,train_label_num,train_path_length = get_data('cooking.train')
    test_data, test_label,test_counts,test_label_num,test_path_length = get_data('cooking.valid')
    train_counts.extend(test_counts)
    counts = [train_counts.count(item) for item in set(train_counts)]
    return train_data, train_label,train_label_num,train_path_length, test_data, test_label,test_label_num,test_path_length, counts





