import jieba
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback
import tensorflow as tf
from collections import Counter
import gensim
import re
import keras.backend as K
import pickle


def fc_bn_ac(input_size, hidden_size, activation='relu'):
    raw_input = Input(shape=(input_size, ))
    fc = Dense(hidden_size)(raw_input)
    fc_bn = BatchNormalization()(fc)
    fc_bn_ac = Activation(activation)(fc_bn)
    return Model(raw_input, fc_bn_ac)


def word_tokenize(content):
    re_han_match = re.compile('[\u4e00-\u9fa5]+')
    clean_content = []
    for i in range(len(content)):
        sentence = ''.join(re_han_match.findall(content[i].strip()))
        clean_content.append(jieba.lcut(sentence, cut_all=False))
    return clean_content


def sent_tokenize(content):
    re_han_match = re.compile('[\u4e00-\u9fa5]+')
    re_sent_tokenize = re.compile(r'[。.！!？?;；~～\n]+')
    clean_content = []
    for i in range(len(content)):
        sents = re_sent_tokenize.split(content[i].strip())
        clean_sents = []
        for i in range(len(sents)):
            if sents[i].strip():
                sent = ''.join(re_han_match.findall(sents[i].strip()))
                clean_sents.append(jieba.lcut(sent, cut_all=False))
        clean_content.append(clean_sents)
    return clean_content


def get_f1_score(y_true, y_pred):
    scores = []
    scores.append(f1_score(y_true, y_pred, labels=[0], average='macro'))
    scores.append(f1_score(y_true, y_pred, labels=[1], average='macro'))
    scores.append(f1_score(y_true, y_pred, labels=[2], average='macro'))
    scores.append(f1_score(y_true, y_pred, labels=[3], average='macro'))
    scores.append(np.mean(scores))
    return scores


def load_pretrained_embedding_matrix():
    tokenizer = pickle.load(open('tokenizer_3W.pkl', 'rb'))
    word_index = tokenizer.word_index
    f = open('sgns.weibo.bigram-char', 'rb')
    f.readline()
    embedding_dict = dict()
    max_nb_words = 30000
    embed_dim = 300
    embedding_matrix = np.zeros((max_nb_words, embed_dim))
    for line in f.readlines():
        line = line.decode(encoding='utf-8').strip()
        word, vec = line.split(' ')[0], [float(x) for x in line.split(' ')[1:]]
        embedding_dict[word] = vec

    for word, i in word_index.items():
        if i > max_nb_words:
            break
        vec = embedding_dict.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    return embedding_matrix



def single_f1(interested_id, id_true, id_pred):
    precison_mask = K.cast(K.equal(id_pred, interested_id), 'float32')
    recall_mask = K.cast(K.equal(id_true, interested_id), 'float32')
    tp = K.cast(K.equal(id_true, id_pred), 'float32') * precison_mask
    precision = K.sum(tp)/(K.sum(precison_mask) + K.epsilon())
    recall = K.sum(tp)/(K.sum(recall_mask) + K.epsilon())
    return K.variable(2.0) * (precision * recall)/(precision + recall + K.epsilon())


def f1_metrics(y_true, y_pred):
    id_true = K.argmax(y_true, axis=-1)
    id_pred = K.argmax(y_pred, axis=-1)
    f1_list = []
    for id in range(4):
        f1_list.append(single_f1(id, id_true, id_pred))
    f1_tensor = K.stack(f1_list)
    return K.mean(f1_tensor)

def f1_0(y_true, y_pred):
    id_true = K.argmax(y_true, axis=-1)
    id_pred = K.argmax(y_pred, axis=-1)
    return single_f1(0, id_true, id_pred)

def f1_1(y_true, y_pred):
    id_true = K.argmax(y_true, axis=-1)
    id_pred = K.argmax(y_pred, axis=-1)
    return single_f1(1, id_true, id_pred)


def f1_2(y_true, y_pred):
    id_true = K.argmax(y_true, axis=-1)
    id_pred = K.argmax(y_pred, axis=-1)
    return single_f1(2, id_true, id_pred)


def f1_3(y_true, y_pred):
    id_true = K.argmax(y_true, axis=-1)
    id_pred = K.argmax(y_pred, axis=-1)
    return single_f1(3, id_true, id_pred)


def create_vocabulary_index(content_vocab):
    corpus = []
    for i in range(len(content_vocab)):
        corpus += content_vocab[i]
    counter = Counter(corpus)
    word2index = dict()
    index2word = dict()
    word2index['pad_word'] = 0
    index2word[0] = 'pad_word'
    index = 1
    for word, count in counter.most_common():
        punctuation = set(r'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
        if word not in punctuation:
            word2index[word] = index
            index2word[index] = word
            index += 1
    return index, index2word, word2index


def create_content_index(content_vocab, word2index):
    content_index = []
    for i in range(len(content_vocab)):
        sentence_index = []
        for word in content_vocab[i]:
            index = word2index.get(word, 0)
            sentence_index.append(index)
        content_index.append(sentence_index)
    return content_index


def assign_word_embedding(sess, index2word, vocab_size, fast_text, embed_size):
    embedding_model_path = 'model_s_300_w_5.bin'
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_model_path, binary=True)
    embedding_dict = [[]] * vocab_size
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    embedding_dict[0] = np.zeros(embed_size)
    for i in range(1, vocab_size):
        word = index2word[i]
        embedding = None
        try:
            embedding = embedding_model[word]
        except Exception:
            embedding = None
        if embedding is not None:
            embedding_dict[i] = embedding
        else:
            embedding_dict[i] = np.random.uniform(-bound, bound, embed_size)

    word_embedding = tf.constant(np.array(embedding_dict), dtype=tf.float32)
    assign_embedding = tf.assign(fast_text.embedding_dict, word_embedding)
    sess.run(assign_embedding)
