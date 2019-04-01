import numpy as np
import pandas as pd
import pickle
import keras.backend as K
import tensorflow as tf
from keras.models import load_model
from keras.layers import *
import config
from keras.preprocessing.sequence import pad_sequences
from utils import get_f1_score, f1_0, f1_1, f1_2, f1_3, f1_metrics
import time
import copy

maxlen = 350
k_value = 3




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
        attention_logits = K.sum(context_similarity, axis=2)
        attention_logits_max = K.max(attention_logits, axis=1, keepdims=True)
        p_attention = K.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = K.expand_dims(p_attention, axis=2)
        representation = inputs[0] * p_attention_expanded
        representation = K.sum(representation, axis=1)
        return representation

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][2]


custom_dict = dict()
custom_dict['f1_0'] = f1_0
custom_dict['f1_1'] = f1_1
custom_dict['f1_2'] = f1_2
custom_dict['f1_3'] = f1_3
custom_dict['f1_metrics'] = f1_metrics
fast = load_model('fastText_f1_0.5891611469366432.h5', custom_objects=custom_dict)
cnn = load_model("textCNN_f1_0.6746724204042697.h5", custom_objects=custom_dict)
custom_dict['AttentionLayer'] = AttentionLayer
han = load_model('han_f1_0.6898659028466435.h5', custom_objects=custom_dict)
custom_dict['tf'] = tf
custom_dict['k_value'] = k_value
rcnn = load_model('rcnn_f1_0.6857842051984384.h5', custom_objects=custom_dict)
rnn = load_model('rnn_f1_0.6925140704103242.h5', custom_objects=custom_dict)

validate_df = pd.read_csv(config.validatePath, header=0, encoding='utf-8')
x_val = pickle.load(open('word_indices_val.pkl', 'rb'))
x_val = pad_sequences(x_val, maxlen)
y_val = np.asarray(validate_df.iloc[:, 4] + 2)


start = time.time()
fast_pred = fast.predict(x_val)
print('fast_time:{}'.format(time.time()-start))
merge_pred = copy.deepcopy(fast_pred)
merge_pred = np.zeros(fast_pred.shape)

start = time.time()
cnn_pred = cnn.predict(x_val)
print('cnn_time:{}'.format(time.time()-start))
merge_pred = np.add(merge_pred, cnn_pred)
print(get_f1_score(y_val, np.argmax(merge_pred, axis=-1)))


start = time.time()
rnn_pred = rnn.predict(x_val)
print('rnn_time:{}'.format(time.time()-start))
merge_pred = np.add(merge_pred, rnn_pred)
print(get_f1_score(y_val, np.argmax(merge_pred, axis=-1)))


start = time.time()
rcnn_pred = rcnn.predict(x_val)
print('rcnn_time:{}'.format(time.time()-start))
merge_pred = np.add(merge_pred, rcnn_pred)
print(get_f1_score(y_val, np.argmax(merge_pred, axis=-1)))







'''
pred = []
for i in range(len(y_val)):
    result = np.asarray([fastText_pred[i], textCNN_pred[i], han_pred[i]])
    result.sort()
    if len(np.unique(result)) == 3:
        pred.append(han_pred[i])
    else:
        pred.append(result[1])

scores = get_f1_score(y_val, pred)
print(scores)

'''