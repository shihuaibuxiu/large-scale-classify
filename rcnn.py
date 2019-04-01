from keras.models import Model, load_model
from keras.layers import *
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import pandas as pd
import numpy as np
import pickle
import config
from utils import get_f1_score, f1_0, f1_1, f1_2, f1_3, f1_metrics
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import time
import tensorflow as tf
import logging
import tqdm


maxlen = 350
max_nb_words = 30000
embed_dim = 300
batch_size = 128
keep_prob = 1.0
epochs = 5
num_per_epoch = 3000
learning_rate = 0.003
lstm_hidden_size = 64
fc_hiddne_size = 1024
num_classes = 4
k_value = 3
decay_rate = 0.99
decay_step = 30
metrics_list = [f1_0, f1_1, f1_2, f1_3, f1_metrics]
DEBUG = True


def preprocessing_data():
    train_df = pd.read_csv(config.trainPath, header=0, encoding='utf-8')
    x_train = pickle.load(open('word_indices_train.pkl', 'rb'))
    x_train = pad_sequences(x_train, maxlen)
    y_train = np.asarray(train_df.iloc[:, 4]) + 2
    y_train_oh = to_categorical(y_train, 4)
    return x_train, y_train, y_train_oh


def data_generator(x_train, y_train_oh):
    data_len = len(y_train_oh)
    permutation = np.random.permutation(data_len)
    x_train = x_train[permutation]
    y_train_oh = y_train_oh[permutation]
    data_id = np.arange(data_len)
    for start, end in zip(data_id[0:data_len:batch_size], data_id[batch_size:data_len:batch_size]):
        yield x_train[start:end], y_train_oh[start:end]


def get_model():
    if DEBUG:
        embedding_matrix = pickle.load(open('embedding_matrix.pkl', 'rb'))

        raw_input = Input(shape=(maxlen, ))
        embedding = Embedding(max_nb_words, embed_dim, weights=[embedding_matrix], trainable=False)(raw_input)
        bi_lstm = Bidirectional(LSTM(units=lstm_hidden_size, return_sequences=True))(embedding)
        fw_lstm, bw_lstm = Bidirectional(LSTM(units=lstm_hidden_size, return_sequences=True), merge_mode=None)(bi_lstm)

        cat = Concatenate(axis=-1)([fw_lstm, embedding, bw_lstm])
        kmaxpooling = Lambda(lambda x: tf.nn.top_k(tf.transpose(x, [0, 2, 1]), k=k_value)[0])(cat)
        flatten = Flatten()(kmaxpooling)
        fc = Dense(fc_hiddne_size)(flatten)
        fc_bn = BatchNormalization(momentum=0.1)(fc)
        fc_ac = Activation('relu')(fc_bn)

        out = Dense(num_classes, activation='sigmoid')(fc_ac)
        model = Model(inputs=raw_input, outputs=out)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=metrics_list)
        return model
    else:
        custom_dict = dict()
        custom_dict['f1_0'] = f1_0
        custom_dict['f1_1'] = f1_1
        custom_dict['f1_2'] = f1_2
        custom_dict['f1_3'] = f1_3
        custom_dict['f1_metrics'] = f1_metrics
        custom_dict['tf'] = tf
        custom_dict['k_value'] = k_value
        return load_model('rcnn.h5', custom_objects=custom_dict)


def train(model, x_train, y_train, y_train_oh):
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    d_class_weights = dict(enumerate(class_weights))
    d_class_weights = {x: 1 for x in range(4)}
    print(d_class_weights)
    step = 1
    #step = pickle.load(open('rcnn_step.pkl', 'rb'))+1
    best_f1 = 0.68
    best_f1 = pickle.load(open('rcnn_f1.pkl', 'rb'))
    print(step, best_f1)

    logging.basicConfig(level=logging.INFO)
    f = open('rcnn_log.txt', 'a')
    for epoch in range(epochs):
        for x, y in tqdm.tqdm(data_generator(x_train, y_train_oh)):
            metrics = model.train_on_batch(x, y)

            log = 'step:{}    num_data:{}    loss:{:.4f}    f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
            logging.info(log.format(step, step * batch_size, *metrics))

            if step % 20 == 0:
                model.save('rcnn.h5')
                pickle.dump(step, open('rcnn_step.pkl', 'wb'))
                print('model has been saved')

            if step % 200 == 0:
                scores = validate(model)
                val_log = 'step:{}    f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
                f.write(val_log.format(step, *scores) + '\n')
                f.flush()
                curr_f1 = scores[-1]
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    model.save('rcnn_f1_{}.h5'.format(best_f1))
                    pickle.dump(best_f1, open('rcnn_f1.pkl', 'wb'))
                    print("it's the best f1:{}".format(best_f1))

            if step % decay_step == 0:
                lr = learning_rate * (decay_rate ** (step // decay_step))
                model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=metrics_list)
                print('curr_lr:{}'.format(lr))

            step += 1


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
    return scores


if __name__ == '__main__':
    x_train, y_train, y_train_oh = preprocessing_data()
    model = get_model()
    model.summary()
    train(model, x_train, y_train, y_train_oh)
    #validate(model)
