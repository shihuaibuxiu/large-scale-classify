import torch
import torch.optim
from torch import nn
import numpy as np
import pickle
import pandas as pd
import config
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import tqdm
import time
from utils import get_f1_score
import logging


log_dir = 'logs/cnn'
kernel_sizes = [2,3,4,5]
num_kernels = 64
embedding_dim = 300
maxlen = 350
max_nb_words = 30000
linear_hidden_size = 1024
num_classes = 4
batch_size = 128
epochs = 4
learning_rate = 0.003
weight_dacay = 1e-5
decay_rate = 0.99
decay_step = 30
embedding_matrix_train = False
update = True


class textCNN(nn.Module):
    def __init__(self):
        super(textCNN, self).__init__()
        self.encoder = nn.Embedding(max_nb_words, embedding_dim)
        embeddding_matrix = torch.Tensor(pickle.load(open('embedding_matrix.pkl', 'rb')))
        self.encoder.weight = nn.Parameter(embeddding_matrix, requires_grad=embedding_matrix_train)

        convs = [nn.Sequential(
                        nn.Conv1d(embedding_dim, num_kernels, kernel_size),
                        nn.BatchNorm1d(num_kernels),
                        nn.ReLU(inplace=True),

                        nn.Conv1d(num_kernels, num_kernels, kernel_size),
                        nn.BatchNorm1d(num_kernels),
                        nn.ReLU(inplace=True),

                        nn.MaxPool1d(maxlen-kernel_size*2+2)
        ) for kernel_size in kernel_sizes]

        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes)*num_kernels, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(linear_hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.long()
        encoder = self.encoder(x)
        convs = [conv(encoder.permute(0, 2, 1)) for conv in self.convs]
        cat = torch.cat(convs, dim=1)
        reshaped = cat.view(cat.size(0), -1)
        logits = self.fc(reshaped)
        return logits


def preprocessing_data(data='train'):
    if data == 'train':
        df = pd.read_csv(config.trainPath, header=0, encoding='utf-8')
        x_train = pickle.load(open('word_indices_train.pkl', 'rb'))
        x_train = pad_sequences(x_train, maxlen)
        y_train = np.asarray(df.iloc[:, 4]) + 2
        return x_train, y_train
    else:
        df = pd.read_csv(config.validatePath, header=0, encoding='utf-8')
        x_val = pickle.load(open('word_indices_val.pkl', 'rb'))
        x_val = pad_sequences(x_val, maxlen)
        y_val = np.asarray(df.iloc[:, 4]) + 2
        return x_val, y_val


def get_optimizer(model,lr1,lr2=0,weight_decay = 0):
    ignored_params = list(map(id, model.encoder.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())
    if lr2 is None:
        lr2 = lr1*0.5
    optimizer = torch.optim.Adam([
            dict(params=base_params,weight_decay = weight_decay,lr=lr1),
            #{'params': model.encoder.parameters(), 'lr': lr2}
        ])
    return optimizer



def validate(model, val_loader):
    model.eval()
    start = time.time()
    scores = []
    for x_val, y_val in val_loader:
        out = model(x_val)
        scores.append(get_f1_score(y_val, torch.argmax(out, dim=1)))
    scores = np.mean(scores, axis=0)
    print(scores)
    print('validation time: ', time.time()-start)
    model.train()
    return scores



if __name__ == '__main__':
    x_train, y_train = preprocessing_data(data='train')
    x_val, y_val = preprocessing_data(data='val')
    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=list(zip(x_val, y_val)), batch_size=batch_size, shuffle=False)

    logging.basicConfig(level=logging.INFO)

    model = textCNN()
    for name, parameters in model.named_parameters():
        print(name, parameters.size())
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr1=learning_rate, weight_decay=weight_dacay)

    old_epoch = 1
    step = 1

    if update:
        checkpoint = torch.load('cnn_ckpt.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        old_epoch = checkpoint['epoch']
        step = checkpoint['step']+1

    best_f1 = 0.67

    writer = SummaryWriter(log_dir=log_dir)
    dummy_input = torch.randint(0, 100, (batch_size, maxlen))
    writer.add_graph(model, dummy_input)


    model.train()
    for epoch in range(old_epoch, epochs+1):
        for data in tqdm.tqdm(train_loader):
            x_train, y_train = data
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()

            log = 'loss:{:.4f}    epoch:{}    step:{}    num_data:{}    ' \
                  'f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
            scores = get_f1_score(y_train, torch.argmax(out, dim=1))
            logging.info(log.format(loss, epoch, step, batch_size*step, *scores))
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('f1_mean', scores[-1], step)
            writer.add_scalars('f1_single', {'f1_0':scores[0], 'f1_1':scores[1], 'f1_2':scores[2], 'f1_3':scores[3]}, step)



            if step % 20 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }, 'cnn_ckpt.tar')
                print('model has been saved')


            if step % 200 == 0:
                scores = validate(model, val_loader)
                scores_log = 'f1_0:{:.4f}    f1_1:{:.4f}    f1_2:{:.4f}    f1_3:{:.4f}    f1_mean:{:.4f}'
                writer.add_text('val_f1', scores_log.format(*scores), step)
                curr_f1 = scores[-1]
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    torch.save(model.state_dict(), 'best_cnn_f1_{}.pt'.format(best_f1))
                    #optimizer = get_optimizer(model, lr1=learning_rate/2, weight_decay=weight_dacay)
                    print("it's the best f1:{}".format(best_f1))


            if step % decay_step == 0:
                lr = learning_rate * (decay_rate ** (step // decay_step))
                optimizer = get_optimizer(model, lr1=lr, weight_decay=weight_dacay)
                print('curr_lr:{}'.format(lr))

            step += 1
    writer.close()



