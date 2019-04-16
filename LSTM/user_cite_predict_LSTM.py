import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import *
from keras.layers import LSTM, Dense, Activation,Dropout
from keras.optimizers import SGD

# load data and preprocessing
def load_data(file_name,test=0.8):
    scale = [0,1,2,3,4,7]
    df = pd.read_csv(file_name, sep=',', usecols=scale)
    data_all = np.array(df).astype(float)
    # scale the data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    data_x = x_scaler.fit_transform(data_all[:, :-1])
    data_y = y_scaler.fit_transform(data_all[:, -1].reshape(-1, 1))
    reshaped_data_x = np.array(data_x).astype('float64')
    reshaped_data_y = np.array(data_y).astype('float64')
    x = reshaped_data_x
    y = reshaped_data_y
    # Seperate the data into training set and the test set
    test_bound=int(reshaped_data_x.shape[0] * test)
    train_x = x[:test_bound]
    test_x=x[test_bound:]
    print('test_bound',test_bound)
    train_y = y[: test_bound]
    test_y = y[test_bound:]

    return train_x, train_y, test_x, test_y, x_scaler, y_scaler

def build_model():
    # input_dim is the last dim of train_x，which is (n_samples, time_steps, input_dim)
    # build the LSTM model based on Keras.
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.5))
    # model.add(LSTM(100, return_sequences=True))
    # model.add(LSTM(1000, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))
    # rms = optimizers.RMSprop(lr=0.001 ,rho=0.9, epsilon=1e-6)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    # model.compile(loss='mse', optimizer=rms)
    return model

#训练
def train_model(train_x, train_y, test_x, test_y):
    model = build_model()
    try:
        # batch_size for optimize
        model.fit(train_x, train_y, batch_size=100, nb_epoch=20, validation_split=0.1)
        # Save the model
        model.save('my_model_accumulate_t1.h5')
        #prediction
        predict_t=model.predict(test_x)
        predict_t=np.reshape(predict_t, (predict_t.size, ))
    except KeyboardInterrupt:
        print(predict_t)
        print(test_y)
    return predict_t,test_y
# calculate MAPE and ACC for evaluation of the LSTM model on the test set.
def evaluation(y_pd,y_true,thd=0.3):#thd=0.1 or 0.3
    print(y_pd)
    print('true',y_true)
    err=(y_pd-y_true)/(y_true)
    abs_err = np.abs(err)
    mape=np.sum(abs_err)/y_true.shape[0]
    acc=np.sum(abs_err<=thd)/y_true.shape[0]

    return mape,acc
# Use the trained LSTM model after training.
def use_mode(file_model,test):
    model=load_model(file_model)
    predict=model.predict(test)
    return predict

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, x_scaler, y_scaler= load_data('output_accumulate.csv')
    print(train_x.shape)
    print(train_y.shape)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))#设置成3维
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    predict_t, test_y = train_model(train_x, train_y, test_x, test_y)
    # Use the trained LSTM model after training.
    # predict_t=use_mode(test_x)
    predict_t = y_scaler.inverse_transform(predict_t.reshape(-1, 1))
    test_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))
    print('reshape_predict:', predict_t)
    print('reshape_test_y:', test_y)
    f = open('./test.txt', 'w')
    f.write(str(predict_t))
    f.close()
    # Calculate the MAPE and ACC
    mape,acc=evaluation(predict_t,test_y)
    print('MAPE: ',mape)
    print('ACC: ',acc)


