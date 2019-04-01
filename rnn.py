from keras.models import Model, load_model
from keras.layers import *
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorboardX import FileWriter
import pickle
import config
from utils import get_f1_score, f1_0, f1_1, f1_2, f1_3, f1_metrics
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam
import time
import tqdm
import logging


log_dir='logs/rnn'
lstm_hidden_size = 64
maxlen = 350
emb_dim = 300
max_nb_words = 30000
learning_rate = 0.003
keep_prob = 1.0
epochs = 3
batch_size = 128
num_per_epoch = 300
k_value = 3
decay_rate = 0.99
decay_step = 30
metrics_list = [f1_0, f1_1, f1_2, f1_3, f1_metrics]
DEBUG = False


def data_preprocessing():
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

        raw_input = Input(shape=(maxlen,))
        embedding = Embedding(input_dim=max_nb_words, output_dim=emb_dim, weights=[embedding_matrix], trainable=False)(raw_input)
        lstm_1 = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))(embedding)
        lstm_2 = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))(lstm_1)
        kmaxpooling = Lambda(lambda x: tf.nn.top_k(tf.transpose(x, [0, 2, 1]), k=k_value)[0])(lstm_2)
        flatten = Flatten()(kmaxpooling)

        fc = Dense(1024)(flatten)
        fc_bn = BatchNormalization(momentum=0.1)(fc)
        fc_ac = Activation('relu')(fc_bn)
        out = Dense(4, activation='sigmoid')(fc_ac)
        model = Model(inputs=raw_input, outputs=out)
        metrics_list = [f1_0, f1_1, f1_2, f1_3, f1_metrics]
        model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=metrics_list)
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
        return load_model('rnn.h5', custom_objects=custom_dict)



def train(model, x_train, y_train, y_train_oh):
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    d_class_weights = dict(enumerate(class_weights))
    d_class_weights = {x: 1 for x in range(4)}
    print(d_class_weights)
    step = 1
    step = pickle.load(open('rnn_step.pkl', 'rb'))+1
    best_f1 = 0.69
    #best_f1 = pickle.load(open('rnn_f1.pkl', 'rb'))
    print(step, best_f1)

    logging.basicConfig(level=logging.INFO)

    sess = tf.Session()
    writer = tf.summary.FileWriter(logdir=log_dir, session=sess)
    loss = tf.placeholder(tf.float32, [])
    f1_mean = tf.placeholder(tf.float32, [])
    f1_single_0 = tf.placeholder(tf.float32, [])
    f1_single_1 = tf.placeholder(tf.float32, [])
    f1_single_2 = tf.placeholder(tf.float32, [])
    f1_single_3 = tf.placeholder(tf.float32, [])
    loss_summary = tf.summary.scalar('loss', loss)
    f1_0_summary = tf.summary.scalar('f1_0', f1_single_0)
    f1_1_summary = tf.summary.scalar('f1_1', f1_single_1)
    f1_2_summary = tf.summary.scalar('f1_2', f1_single_2)
    f1_3_summary = tf.summary.scalar('f1_3', f1_single_3)
    f1_mean_summary = tf.summary.scalar('f1_mean', f1_mean)

    merged_op = tf.summary.merge_all()
    #tf.gfile.DeleteRecursively(log_dir)


    for epoch in range(epochs):
        for x, y in tqdm.tqdm(data_generator(x_train, y_train_oh)):
            metrics = model.train_on_batch(x, y)
            log = 'step:{}    num_data:{}    loss:{:.4f}    f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
            summary = sess.run(merged_op, feed_dict={loss:metrics[0], f1_single_0:metrics[1], f1_single_1:metrics[2],
                                                     f1_single_2:metrics[3], f1_single_3:metrics[4], f1_mean:metrics[-1]})
            writer.add_summary(summary, step)

            logging.info(log.format(step, step*batch_size, *metrics))



            if step % 20 == 0:
                model.save('rnn.h5')
                pickle.dump(step, open('rnn_step.pkl', 'wb'))
                print('model has been saved')


            if step % 200 == 0:
                scores = validate(model)
                curr_f1 = scores[-1]
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    model.save('rnn_f1_{}.h5'.format(best_f1))
                    print("it's the best f1:{}".format(best_f1))

            if step % decay_step == 0:
                lr = learning_rate * (decay_rate ** (step // decay_step))
                model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=metrics_list)
                print('curr_lr:{}'.format(lr))

            step += 1



    '''
    decay_rate = 0.96
    decay_step = 5
    while step < 140:
        if step % decay_step == 0:
            lr = learning_rate * (decay_rate ** (step//decay_step))
            model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=metrics_list)
            print('curr_lr:{}'.format(lr))
        permutation = np.random.permutation(len(y_train))[:num_per_epoch]
        batch_x = x_train[permutation]
        batch_y = y_train_oh[permutation]
        #history = model.fit(batch_x, batch_y, batch_size=batch_size, epochs=epochs, shuffle=False, class_weight=d_class_weights)
        history = model.train_on_batch()
        print(type(history.history))
        print(history.history['loss'])
        step += 1
        print('step:{0}   num_train_data:{1}'.format(step, step*num_per_epoch))
        model.save('rnn.h5')
        pickle.dump(step, open('rnn_step.pkl', 'wb'))
        if step%5 == 0:
            curr_f1 = validate(model)
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                model.save('rnn_f1_{}.h5'.format(best_f1))
                pickle.dump(best_f1, open('rnn_f1.pkl', 'wb'))
    '''


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
    x_train, y_train, y_train_oh = data_preprocessing()
    model = get_model()
    model.summary()
    train(model, x_train, y_train, y_train_oh)
    #validate(model)


