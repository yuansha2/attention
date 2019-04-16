import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.layers.core import *
from keras.optimizers import *
from scipy.stats import pareto
from attention_utils import get_activations, get_data_recurrent
import sklearn.metrics
from keras.layers import merge
from keras.layers.recurrent import LSTM


def load_data(file_name,test=0.8):
    # scale = list(range(6))
    scale = [0,1,2,3,4,5]
    df = pd.read_csv(file_name, sep=',', usecols=scale)
    # df = pd.read_csv(file_name, sep=',')
    # print(df[:5])
    # data_all_1 = np.array(df).astype(float)
    # np.random.shuffle(data_all_1)
    # data_all_shuffled = data_all_1
    # data_all_shuffled = pd.DataFrame(data_all_shuffled)
    # data_all = data_all_shuffled.iloc[:,scale]
    data_all = np.array(df).astype(float)
    # print(data_all_shuffled[:5])
    # data_all_shuffled.to_csv('./data_new.csv',index=False,header=True)
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
    print('test_bound',test_bound)
    train_y = y[: test_bound]
    test_y = y[test_bound:]

    return train_x, train_y, test_x, test_y


INPUT_DIM = 1
TIME_STEPS = 5
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 100
    # drop1 = Dropout(0.3)(inputs)
    # lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(inputs)
    # lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    # lstm_out = LSTM(100, return_sequences=True)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    # output = Dense(1, activation='linear')(attention_mul)
    output = Dense(1, activation='linear')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


#训练
def train_model(train_x, train_y, test_x, test_y):
    model = model_attention_applied_after_lstm()
    try:
        # batch_size for optimize
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        model.compile(optimizer='adadelta', loss='mape', metrics=['accuracy'])
        model.fit(train_x, train_y, batch_size=100, nb_epoch=50, validation_split=0.1)
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

    train_x, train_y, test_x, test_y= load_data('output_accumulate.csv')
    print(train_x.shape)
    print(train_y.shape)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))#设置成3维
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    # predict_t=use_mode(test_x)
    #以下是训练过程
    predict_t, test_y = train_model(train_x, train_y, test_x, test_y)
    # predict_t=use_mode("my_model_accumulate_t1.h5",test_x)
    predict_t = predict_t.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    print('reshape_predict:', predict_t)
    print('reshape_test_y:', test_y)
    f = open('./test.txt', 'w')
    f.write(str(predict_t))
    f.close()

    #最后会有一个“nonetype”的错误，暂时不用管。可能跟cpu使用有关
    # caculate indication
    predict_t_round = np.round(predict_t)
    # mape,acc=evaluation(predict_t,test_y)
    mape, acc = evaluation(predict_t_round, test_y)
    print('MAPE: ',mape)
    print('ACC: ',acc)

