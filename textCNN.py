from keras.models import Model, load_model
from keras.layers import *
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import config
from utils import get_f1_score, f1_0, f1_1, f1_2, f1_3, f1_metrics
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam
import time
import tensorflow as tf
import logging
import tqdm


log_dir = 'logs/cnn'
filter_sizes=[2,3,4,5]
num_filters = 64
k_value = 3
maxlen = 350
emb_dim = 300
max_nb_words = 30000
learning_rate = 0.003
keep_prob = 1.0
epochs = 5
batch_size = 128
num_per_epoch = 3000
DEBUG = False
metrics_list = [f1_0, f1_1, f1_2, f1_3, f1_metrics]
decay_rate = 0.99
decay_step = 30



def data_prepocessing():
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

        sentence_indices = Input(shape=(maxlen,))
        embedding = Embedding(input_dim=max_nb_words, output_dim=emb_dim, weights=[embedding_matrix], trainable=False)(sentence_indices)
        outputs_cnn = []
        for filter_size in filter_sizes:
            conv1D = Conv1D(num_filters, filter_size)(embedding)
            conv1D_bn = BatchNormalization(momentum=0.1)(conv1D)
            conv1D_ac = Activation('relu')(conv1D_bn)

            conv1D_2 = Conv1D(num_filters, filter_size)(conv1D_ac)
            conv1D_2_bn = BatchNormalization(momentum=0.1)(conv1D_2)
            conv1D_2_ac = Activation('relu')(conv1D_2_bn)

            conv1D_3 = Conv1D(num_filters, filter_size)(conv1D_2_ac)
            conv1D_3_bn = BatchNormalization(momentum=0.1)(conv1D_3)
            conv1D_3_ac = Activation('relu')(conv1D_3_bn)

            conv1D_4 = Conv1D(num_filters, filter_size)(conv1D_3_ac)
            conv1D_4_bn = BatchNormalization(momentum=0.1)(conv1D_4)
            conv1D_4_ac = Activation('relu')(conv1D_4_bn)

            #pool1D = GlobalMaxPool1D()(conv1D_2_ac)
            kmaxpooling = Lambda(lambda x: tf.nn.top_k(tf.transpose(x, [0, 2, 1]), k=k_value)[0])(conv1D_4_ac)
            flatten = Flatten()(kmaxpooling)
            outputs_cnn.append(flatten)
        concat = Concatenate(axis=-1)(outputs_cnn)
        fc = Dense(1024)(concat)
        fc_bn = BatchNormalization(momentum=0.1)(fc)
        fc_ac = Activation('relu')(fc_bn)
        out = Dense(4, activation='sigmoid')(fc_ac)
        model = Model(inputs=sentence_indices, outputs=out)
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
        return load_model('cnn.h5', custom_objects=custom_dict)



def train(model, x_train, y_train, y_train_oh):
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    d_class_weights = dict(enumerate(class_weights))
    d_class_weights = {x: 1 for x in range(4)}
    print(d_class_weights)
    step = 1
    step = pickle.load(open('cnn_step.pkl', 'rb'))+1
    best_f1 = 0.67
    best_f1 = pickle.load(open('cnn_f1.pkl', 'rb'))
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
    text = tf.placeholder(tf.string, [])
    text_summary = tf.summary.text('val_f1', text)
    # tf.gfile.DeleteRecursively(log_dir)


    f = open('cnn_log.txt', 'a')
    for epoch in range(epochs):
        for x, y in tqdm.tqdm(data_generator(x_train, y_train_oh)):
            metrics = model.train_on_batch(x, y)
            summary = sess.run(merged_op, feed_dict={loss: metrics[0], f1_single_0: metrics[1], f1_single_1: metrics[2],
                                                     f1_single_2: metrics[3], f1_single_3: metrics[4],
                                                     f1_mean: metrics[-1]})
            writer.add_summary(summary, step)

            log = 'step:{}    num_data:{}    loss:{:.4f}    f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
            logging.info(log.format(step, step * batch_size, *metrics))
            f.write(log.format(step, step * batch_size, *metrics) + '\n')
            f.flush()

            if step % 20 == 0:
                model.save('cnn.h5')
                pickle.dump(step, open('cnn_step.pkl', 'wb'))
                print('model has been saved')

            if step % 200 == 0:
                scores = validate(model)
                val_log = 'f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
                val_text = sess.run(text_summary, feed_dict={text:val_log.format(*scores)})
                writer.add_summary(val_text, step)
                curr_f1 = scores[-1]
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    model.save('cnn_f1_{}.h5'.format(best_f1))
                    pickle.dump(best_f1, open('cnn_f1.pkl', 'wb'))
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
    x_train, y_train, y_train_oh = data_prepocessing()
    model = get_model()
    model.summary()
    #train(model, x_train, y_train, y_train_oh)
    #validate(model)


