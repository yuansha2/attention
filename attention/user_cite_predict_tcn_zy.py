#source_code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import *
from keras.layers import LSTM, Dense, Activation,Dropout
from keras.optimizers import SGD
import tcn

def load_data(file_name,test=0.8):
    scale = [0,1,2,3,4,7]
    df = pd.read_csv(file_name, sep=',', usecols=scale)
    data_all = np.array(df).astype(float)
    # 是否对输入集进行scale
    # x_scaler = MinMaxScaler()
    # y_scaler = MinMaxScaler()
    # data_x = x_scaler.fit_transform(data_all[:, :-1])
    # data_y = y_scaler.fit_transform(data_all[:, -1].reshape(-1, 1))
    data_x = data_all[:, :-1]
    data_y = data_all[:, -1].reshape(-1, 1)
    reshaped_data_x = np.array(data_x).astype('float64')
    reshaped_data_y = np.array(data_y).astype('float64')
    x = reshaped_data_x
    y = reshaped_data_y
    #划分！最后要做learning curve就不需要cv这个了，用模型自带的做图
    test_bound=int(reshaped_data_x.shape[0] * test)
    train_x = x[:test_bound]
    test_x=x[test_bound:]

    train_y = y[: test_bound]
    test_y = y[test_bound:]

    # return train_x, train_y, test_x, test_y, x_scaler, y_scaler
    return train_x, train_y, test_x, test_y

def build_model():
    model = tcn.dilated_tcn(output_slice_index='last',
                            num_classes=0,
                            num_feat=1,
                            nb_filters=5,
                            kernel_size=5,
                            dilatations=[1, 2, 4, 8],
                            nb_stacks=8,
                            max_len=5,
                            activation='linear',
                            regression=True)
    return model

#训练
def train_model(train_x, train_y, test_x, test_y):
    model = build_model()
    try:
        # batch_size for optimize
        model.fit(train_x, train_y, batch_size=32, nb_epoch=50, validation_split=0.1)
        #用于保存模型以便事后加载
        model.save('my_model_accumulate_t1.h5')
        #prediction
        predict_t=model.predict(test_x)
        predict_t=np.reshape(predict_t, (predict_t.size, ))
    except KeyboardInterrupt:
        print(predict_t)
        print(test_y)
    return predict_t,test_y
#mape 以及acc
def evaluation(y_pd,y_true,thd=0.3):#thd=0.1 更为合理
    print(y_pd)
    print('true',y_true)
    err=(y_pd-y_true)/(y_true)
    abs_err = np.abs(err)
    mape=np.sum(abs_err)/y_true.shape[0]
    acc=np.sum(abs_err<=thd)/y_true.shape[0]

    return mape,acc
#训练好后可以直接加载
def use_mode(file_model,test):
    model=load_model(file_model)
    predict=model.predict(test)
    return predict

if __name__ == '__main__':


    # train_x, train_y, test_x, test_y, x_scaler, y_scaler= load_data('output_accumulate.csv')
    train_x, train_y, test_x, test_y = load_data('output_accumulate.csv')
    print(train_x.shape)
    print(train_y.shape)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))#设置成3维
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    #以下是训练过程
    predict_t, test_y = train_model(train_x, train_y, test_x, test_y)
    # predict_t=use_mode("my_model_accumulate_t2_100_1000.h5",test_x)
    # predict_t = y_scaler.inverse_transform(predict_t.reshape(-1, 1))
    # test_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))
    print('reshape_predict:', predict_t)
    print('reshape_test_y:', test_y)
    f = open('./test.txt', 'w')
    f.write(str(predict_t))
    f.close()

    # caculate indication
    mape,acc=evaluation(predict_t,test_y)
    print('MAPE: ',mape)
    print('ACC: ',acc)


