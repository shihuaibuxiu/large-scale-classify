from keras.models import Model, load_model
from keras.layers import *
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import config
from utils import get_f1_score, f1_0, f1_1, f1_2, f1_3, f1_metrics
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import time
import tensorflow as tf
import tqdm
import logging


maxlen = 350
max_nb_words = 30000
DEBUG = True
embed_size = 300
ngram_range = 3
batch_size = 128
num_classes = 4
keep_prob = 1.0
epochs = 20
learning_rate = 0.003
metrics_list = [f1_0, f1_1, f1_2, f1_3, f1_metrics]
decay_rate = 0.99
decay_step = 50


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

    def my_loss(y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.nce_loss(
                weights=tf.Variable(tf.truncated_normal((4, embed_size))),
                biases=tf.zeros((4,)),
                num_classes=4,
                num_sampled=4,
                inputs=average_pool,
                labels=tf.expand_dims(tf.argmax(y_true, axis=1), axis=1)))


    if DEBUG:
        embedding_matrix = pickle.load(open('embedding_matrix.pkl', 'rb'))

        raw_input = Input(shape=(maxlen, ))
        embedding = Embedding(max_nb_words, embed_size, weights=[embedding_matrix], trainable=True)(raw_input) #,
        pre_fc = Dense(embed_size*2)(embedding)
        pre_fc = BatchNormalization(momentum=0.1)(pre_fc)
        pre_fc = Activation('relu')(pre_fc)
        average_pool = GlobalAveragePooling1D()(pre_fc)
        #out1 = Dense(4, name='out1')(average_pool)

        post_fc = Dense(embed_size*4)(average_pool)
        post_fc = BatchNormalization(momentum=0.1)(post_fc)
        post_fc = Activation('relu')(post_fc)

        fc_2 = Dense(128)(post_fc)
        fc_2 = BatchNormalization(momentum=0.1)(fc_2)
        fc_2 = Activation('relu')(fc_2)
        out2 = Dense(4, activation='sigmoid', name='out2')(fc_2)
        model = Model(inputs=raw_input, outputs=out2)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=metrics_list)
        return model
    else:
        custom_dict = dict()
        custom_dict['f1_0'] = f1_0
        custom_dict['f1_1'] = f1_1
        custom_dict['f1_2'] = f1_2
        custom_dict['f1_3'] = f1_3
        custom_dict['f1_metrics'] = f1_metrics
        return load_model('fastText.h5', custom_objects=custom_dict)


def train(model, x_train, y_train, y_train_oh):
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    d_class_weights = dict(enumerate(class_weights))
    d_class_weights = {i:1 for i in range(4)}
    step = 1
    #step = pickle.load(open('fast_step.pkl', 'rb'))+1
    best_f1 = 0.58
    best_f1 = pickle.load(open('fast_f1.pkl', 'rb'))
    print(d_class_weights)
    print(step, best_f1)

    logging.basicConfig(level=logging.INFO)
    f = open('fast_log.txt', 'a')
    for epoch in range(epochs):
        for x, y in tqdm.tqdm(data_generator(x_train, y_train_oh)):
            metrics = model.train_on_batch(x, y)

            log = 'step:{}    num_data:{}    loss:{:.4f}    f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
            logging.info(log.format(step, step * batch_size, *metrics))

            if step % 20 == 0:
                model.save('fast.h5')
                pickle.dump(step, open('fast_step.pkl', 'wb'))
                print('model has been saved')

            if step % 200 == 0:
                scores = validate(model)
                val_log = 'step:{}    f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
                f.write(val_log.format(step, *scores) + '\n')
                f.flush()
                curr_f1 = scores[-1]
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    model.save('fast_f1_{}.h5'.format(best_f1))
                    pickle.dump(best_f1, open('fast_f1.pkl', 'wb'))
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
    pred = np.argmax(model.predict(x_val), axis=-1)
    scores = get_f1_score(y_val, pred)
    print(scores)
    print('validation time:{}'.format(time.time()-start))
    return scores


if __name__ =='__main__':
    x_train, y_train, y_train_oh = preprocessing_data()
    model = get_model()
    model.summary()
    train(model, x_train, y_train, y_train_oh)
    #validate(model)

