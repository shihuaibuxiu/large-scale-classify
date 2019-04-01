from keras.layers import *
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os
import pickle
import logging
import config
from keras.models import Model, load_model
import keras.backend as K
import tensorflow as tf
from utils import get_f1_score, f1_0, f1_1, f1_2, f1_3, f1_metrics, fc_bn_ac
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam
import time



hidden_size = 128
maxlen = 700
num_sents = 10
sentence_len = maxlen//num_sents
max_nb_words = 30000
embed_dim = 300
keep_prob = 1.0
DEBUG = True
load_original_data = False
num_per_epoch = 3000
batch_size = 128
epochs = 1
decay_rate = 0.96
decay_step = 5
learning_rate = 0.003


def preprocessing_data():
    train_df = pd.read_csv(config.trainPath, header=0, encoding='utf-8')
    x_train = pickle.load(open('word_indices_train.pkl', 'rb'))
    x_train = pad_sequences(x_train, maxlen)
    y_train = np.asarray(train_df.iloc[:,4] + 2)
    y_train_oh = to_categorical(y_train, 4)
    return x_train, y_train, y_train_oh


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1][-1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        hidden_representation = inputs[1]
        context_similarity = hidden_representation * self.W
        attention_logits = K.sum(context_similarity, axis=2)   #step1:点积求和
        attention_logits_max = K.max(attention_logits, axis=1, keepdims=True)
        p_attention = K.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = K.expand_dims(p_attention, axis=2)    #step2: softmax 概率归一化
        representation = inputs[0] * p_attention_expanded
        representation = K.sum(representation, axis=1)  #step3: 加权求和
        return representation

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][2]


def get_model():
    if DEBUG:
        embedding_matrix = pickle.load(open('embedding_matrix.pkl', 'rb'))

        word_input = Input(shape=(sentence_len, embed_dim))
        l_lstm = Bidirectional(GRU(hidden_size, return_sequences=True))(word_input)
        word_hidden_representation = TimeDistributed(fc_bn_ac(hidden_size*2, hidden_size*2))(l_lstm)
        l_att = AttentionLayer()([l_lstm, word_hidden_representation])
        sent_encoder = Model(word_input, l_att)
        sent_encoder.summary()

        sentence_input = Input(shape=(maxlen,))
        embedding = Embedding(max_nb_words, embed_dim, weights=[embedding_matrix], trainable=False)(sentence_input)
        input_reshape = Reshape((num_sents, sentence_len, embed_dim))(embedding)
        review_encoder = TimeDistributed(sent_encoder)(input_reshape)
        l_lstm_sent = Bidirectional(GRU(hidden_size, return_sequences=True))(review_encoder)
        sent_hidden_representation = TimeDistributed(fc_bn_ac(hidden_size*2, hidden_size*2))(l_lstm_sent)
        l_att_sent = AttentionLayer()([l_lstm_sent, sent_hidden_representation])

        fc = Dense(1024)(l_att_sent)
        fc_bn = BatchNormalization()(fc)
        fc_ac = Activation('relu')(fc_bn)
        preds = Dense(4, activation='sigmoid')(fc_ac)
        model = Model(sentence_input, preds)
        metrics_list = [f1_0, f1_1, f1_2, f1_3, f1_metrics]
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=metrics_list)
        return model
    else:
        custom_dict = dict()
        custom_dict['f1_0'] = f1_0
        custom_dict['f1_1'] = f1_1
        custom_dict['f1_2'] = f1_2
        custom_dict['f1_3'] = f1_3
        custom_dict['f1_metrics'] = f1_metrics
        custom_dict['AttentionLayer'] = AttentionLayer
        return load_model('han.h5',  custom_objects=custom_dict)

def train(model, x_train, y_train, y_train_oh):
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    d_class_weights = dict(enumerate(class_weights))
    d_class_weights = {x: 1 for x in range(4)}
    print(d_class_weights)
    step = 0
    #step = pickle.load(open('han_step.pkl', 'rb'))
    best_f1 = 0.67
    #best_f1 = pickle.load(open('han_f1.pkl', 'rb'))
    print(step, best_f1)
    metrics_list = [f1_0, f1_1, f1_2, f1_3, f1_metrics]
    while step < 140:
        if step % decay_step == 0:
            lr = learning_rate * (decay_rate ** (step // decay_step))
            model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=metrics_list)
            print('curr_lr:{}'.format(lr))
        permutation = np.random.permutation(len(y_train))[:num_per_epoch+64]
        batch_x = x_train[permutation[:num_per_epoch]]
        batch_y = y_train_oh[permutation[:num_per_epoch]]

        model.fit(batch_x, batch_y, batch_size=batch_size, epochs=epochs, shuffle=False, class_weight=d_class_weights)
        step += 1
        print('step:{0}   num_train_data:{1}'.format(step, step*num_per_epoch))

        pickle.dump(step, open('han_step.pkl', 'wb'))
        model.save('han.h5')
        if step%5 == 0:
            curr_f1 = validate(model)
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                model.save('han_f1_{}.h5'.format(best_f1))
                pickle.dump(best_f1, open('han_f1.pkl', 'wb'))


def validate(model):
    start = time.time()
    validate_df = pd.read_csv(config.validatePath, header=0, encoding='utf-8')
    x_val = pickle.load(open('word_indices_val.pkl', 'rb'))
    x_val = pad_sequences(x_val, maxlen)
    y_val = validate_df.iloc[:,4]+2
    #x_val = x_val[:2000]
    #y_val = y_val[:2000]
    pred = np.argmax(model.predict(x_val), axis=-1)
    scores = get_f1_score(y_val, pred)
    print(scores)
    print('validation_time:{}'.format(time.time()-start))
    return scores[-1]


if __name__=='__main__':
    x_train, y_train, y_train_oh = preprocessing_data()
    model = get_model()
    model.summary()
    train(model, x_train, y_train, y_train_oh)
    #validate(model)

