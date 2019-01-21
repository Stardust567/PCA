"""
ref:
    # 通过上一期序列值预测下一期
    https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    # theory
    https://deeplearning4j.org/lstm.html#long
"""
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix


def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def future_predict(data, num, model, look_back):
    predict = numpy.zeros((len(num), 1))
    for i in num:
        data_mat = numpy.zeros((1, 1, look_back))
        data_mat[0,0,:] = data
        temp_predict = model.predict(data_mat)
        temp_list = []
        count = look_back-1
        while(count != 0):
            temp_list.append(data[-1*count])
            count -= 1
        temp_list.append((temp_predict[-1])[0])
        data = numpy.array(temp_list)
        predict[i,0] = temp_predict
    return predict


def Lstm(data, x_start = 1981, predict_num=10, look_back = 2):
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset

    # df = read_csv('../dataset/SZIndex.csv',header=-1)
    # dataset = df[6].values
    # dataset = dataset.reshape(dataset.shape[0], 1)

    dataset = numpy.array(data).reshape(-1, 1)
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = len(dataset)
    train = dataset[0:train_size, :]
    # reshape into X=t and Y=t+1
    X,Y = create_dataset(dataset, look_back)
    trainX, trainY = create_dataset(train, look_back)
    # reshape input to be [samples, time steps, features]
    X = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions

    trainPredict = model.predict(trainX)
    print((X[-3])[0])
    dataPredict = future_predict((X[-3])[0], range(predict_num), model, look_back=look_back)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])

    dataPredict = scaler.inverse_transform(dataPredict)
    # calculate root mean squared error 结合RMSE（均方根误差）计算损失
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.zeros((len(dataset)+look_back, 1))
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift data predictions for plotting
    dataPredictPlot = numpy.zeros((len(dataset)+predict_num, 1))
    dataPredictPlot[:, :] = numpy.nan
    dataPredictPlot[len(dataset)-1:len(dataset) + predict_num -1, :] = dataPredict
    # plot baseline and predictions
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(dataPredictPlot)
    plt.legend(('Data', 'Train', 'Predict'),
               loc='best')
    plt.xlim((-1, len(dataset) + predict_num)) # 设置坐标轴范围
    my_x_ticks = numpy.arange(0, len(dataset) + predict_num -1, 5)
    my_x_labels = numpy.arange(x_start, x_start+len(dataset) + predict_num - 1, 5)
    plt.xticks(my_x_ticks, my_x_labels, rotation=30) # 设置坐标轴刻度 & 旋转
    plt.show()

