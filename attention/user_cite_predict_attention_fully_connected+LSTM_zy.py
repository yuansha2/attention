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
from collections import Counter


def load_data(file_name,test=0.8):
    # scale = list(range(6))
    scale = [0,1,2,3,4,9]
    df = pd.read_csv(file_name, sep=',', usecols=scale)
    data_all = np.array(df).astype(float)
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


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    #keras 创建模型的基本方法，可以参见keras的文档
    # model.add(Dense(1, init='uniform', activation='sigmoid'))
    # ,return_sequences=True
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    Dense_1 = Dense(256, input_dim=5, init='uniform', activation='relu')(inputs)
    Dense_2 = Dense(256, init='uniform', activation='relu')(Dense_1)
    Dense_3 = Dense(256, init='uniform', activation='relu')(Dense_2)
    attention_mul = attention_3d_block(Dense_3)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='linear')(attention_mul)
    model = Model(input=[inputs], output=output)
    # rms = optimizers.RMSprop(lr=0.001 ,rho=0.9, epsilon=1e-6)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mse', optimizer=sgd)
    model.compile(loss='mse', optimizer='adam')
    return model


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
    # a = LSTM(32, return_sequences=True)(a)
    a = Dense(TIME_STEPS, activation='linear')(a)
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
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    # lstm_out = LSTM(100, return_sequences=True)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='linear')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 100
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    # output = Dense(1, activation='sigmoid')(attention_mul)
    output = Dense(1, activation='elu')(attention_mul)
    # output = Dense(1)(attention_mul)
    # output = advanced_activations.ThresholdedReLU(theta=1.0)(output)
    model = Model(input=[inputs], output=output)
    return model




#训练
def train_model(train_x, train_y, test_x, test_y):
    # build_model：attention_based_全连接网络
    # model_attention_applied_before_lstm：模型输入层之后加入attention机制
    # model_attention_applied_after_lstm 模型LSTM输出层之后加入attention
    model = build_model()
    # model = model_attention_applied_after_lstm()
    # model = model_attention_applied_before_lstm()
    try:
        # batch_size for optimize
        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        # adadelta = Adadelta(lr=0.001, rho=0.95, epsilon=1e-06)
        # adadelta = Adadelta(lr=0.001, rho=0.95, epsilon=1e-06)
        # model.compile(optimizer='adadelta', loss='mape', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='mape', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.fit(train_x, train_y, batch_size=100, nb_epoch=30, validation_split=0.1)
        #用于保存模型以便事后加载

        model.save('my_model_attention_fully_connect_t1.h5')
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
    # 利用训练好的模型
    # predict_t=use_mode("my_model_accumulate_t1_0.643.h5",test_x)
    predict_t = predict_t.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    predict_t_round = np.round(predict_t)


    # 结果输出
    '''
    result_predict = [int(item) for sublist in predict_t_round for item in sublist]
    result_predict_graph = Counter(result_predict)
    result_predict_paper = list(result_predict_graph.values())
    result_predict_citation = list(result_predict_graph.keys())

    result_GT = [int(item) for sublist in test_y for item in sublist]
    result_GT_graph = Counter(result_GT)
    result_GT_paper = list(result_GT_graph.values())
    result_GT_citation = list(result_GT_graph.keys())

    predict_graph = pd.DataFrame({'result_predict_paper': result_predict_paper, 'citations': result_predict_citation})
    predict_graph = predict_graph.sort_values(by='citations', axis=0, ascending=True)
    GT_graph = pd.DataFrame({'result_GT_paper': result_GT_paper, 'citations': result_GT_citation})
    GT_graph = GT_graph.sort_values(by='citations', axis=0, ascending=True)
    predict_graph.to_csv('./result/predict_graph.csv', index=False, columns=['citations', 'result_predict_paper'])
    GT_graph.to_csv('./result/GT_graph.csv', index=False, columns=['citations', 'result_GT_paper'])

    result_graph = pd.merge(predict_graph, GT_graph, how='outer', on='citations')
    result_graph = result_graph.fillna(0)
    result_graph.to_csv('./result/result_graph.csv', index=False)
    '''

    print('reshape_predict:', predict_t)
    print('reshape_test_y:', test_y)


    mape, acc = evaluation(predict_t_round, test_y)
    print('MAPE: ',mape)
    print('ACC: ',acc)

