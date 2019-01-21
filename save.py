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

def future_predict(data, num, model):
    predict = numpy.zeros((len(num), 1))
    for i in num:
        data = numpy.reshape(data, (1, 1, 2))
        temp_predict = model.predict(data)
        data = numpy.array([((data[-1])[0])[-1], (temp_predict[-1])[0]])
        predict[i,0] = temp_predict
    print(predict)
    return predict


def Lstm(data, predict_num=10):
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
    train_size = int(len(dataset)*0.68) # *0.68
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 2 # 用于推断的历史序列长度
    X,Y = create_dataset(dataset, look_back)
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    X = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    dataPredict = future_predict((X[-1])[0], range(predict_num), model)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    dataPredict = scaler.inverse_transform(dataPredict)
    # calculate root mean squared error 结合RMSE（均方根误差）计算损失
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) -1, :] = testPredict
    # shift data predictions for plotting
    dataPredictPlot = numpy.zeros((len(dataset)+predict_num, 1))
    dataPredictPlot[:, :] = numpy.nan
    dataPredictPlot[len(dataset):len(dataset) + predict_num, :] = dataPredict
    # plot baseline and predictions
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.plot(dataPredictPlot)
    plt.legend(('dataset', 'dataPredictPlot', 'trainPredictPlot', 'testPredictPlot'),
               loc='best')
    plt.show()