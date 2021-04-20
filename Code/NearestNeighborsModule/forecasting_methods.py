import numpy as np
from NearestNeighborsModule.DelayVectorDB import DelayVectorDB
from NearestNeighborsModule.NearestNeighbors import NearestNeighbors
from NearestNeighborsModule.aux_forecast import last_vector
import matplotlib.pyplot as plt
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import pickle


class ForecastingMethods(object):
    def __init__(self, time_series, m, tau, eps, num_forecast, error_fun, dates=None):
        self.__time_series = time_series
        self.__m = m
        self.__tau = tau
        self.__eps = eps
        self.__num_forecast = num_forecast
        self.__error_fun = error_fun
        self.__dates = dates

    def osa(self, test=False):
        validation_set = self.__time_series[-self.__num_forecast:]
        forecast = list()
        time_series_train = self.__time_series[:-self.__num_forecast]
        db_ts = DelayVectorDB(time_series_train, self.__m, self.__tau, dates=self.__dates)
        X, y = db_ts.get_vectors()

        for i in range(self.__num_forecast, 0, -1):
            nn = NearestNeighbors(epsilon=self.__eps)
            nn.fit(X, y)
            new_vector = last_vector(self.__time_series[:-i], [self.__m,], [self.__tau,], dates=self.__dates)
            forecast.append(nn.predict(new_vector))
            X = np.vstack((X, new_vector))
            y = np.vstack((y, [validation_set[np.abs(i - self.__num_forecast)]]))

        forecast = np.array(forecast)
        if test:
            self.plotting(forecast, validation_set)
        return self.__error_fun(validation_set, forecast)

    def one_hour_ahead(self, delta, h, test=False):
        validation_set = self.__time_series[-self.__num_forecast:]
        forecast = list()
        time_series_train = self.__time_series[:-self.__num_forecast]
        db_ts = DelayVectorDB(time_series_train, self.__m, self.__tau, delta=delta, h=h, dates=self.__dates)
        X, y = db_ts.get_vectors()

        for i in range(self.__num_forecast, 0, -1):
            nn = NearestNeighbors(epsilon=self.__eps)
            nn.fit(X, y)
            new_vector = last_vector(self.__time_series[:-i], [self.__m,], [self.__tau,], dates=self.__dates)
            forecast.append(nn.predict(new_vector))
            X = np.vstack((X, new_vector))
            y = np.vstack((y, np.append(y[-1][1:], [validation_set[np.abs(i - self.__num_forecast)]])))
        # todo: test this part more
        # I'm not convinced of this calculation, I think the length of the forecast set should also include h
        forecast = np.array(forecast)[:len(validation_set) - (delta - 1) - 1]

        idx_val = np.array(range(delta))
        db_val = np.array([validation_set[idx_val + i] for i in range(len(validation_set) - (delta - 1))])[h - 1:]
        score = np.mean([self.__error_fun(db_val[:, i], forecast[:, i]) for i in range(delta)])
        if test:
            self.plotting(forecast[:, 0], db_val[:, 0])
        return score  # , forecast, db_val

    def one_day_ahead(self, delta, h, test=False):
        validation_set = self.__time_series[-self.__num_forecast:]
        forecast = list()
        time_series_train = self.__time_series[:-self.__num_forecast]
        db_ts = DelayVectorDB(time_series_train, self.__m, self.__tau, delta=delta, h=h, dates=self.__dates)
        X, y = db_ts.get_vectors()

        for i in range(self.__num_forecast, 0, -delta):
            nn = NearestNeighbors(epsilon=self.__eps)
            nn.fit(X, y)
            new_vector = last_vector(self.__time_series[:-i], [self.__m, ], [self.__tau, ], dates=self.__dates)
            forecast.append(nn.predict(new_vector))
            for j in range(i, i - delta, -1):
                vector = last_vector(self.__time_series[:-j], [self.__m, ], [self.__tau, ], dates=self.__dates)
                X = np.vstack((X, vector))
                y = np.vstack((y, np.append(y[-1][1:], [validation_set[np.abs(j - self.__num_forecast)]])))
        forecast = np.array(forecast).ravel()
        score = self.__error_fun(validation_set, forecast)
        if test:
            self.plotting(forecast, validation_set)
        return score  # , forecast, validation_set

    def nss(self, test=False):
        validation_set = self.__time_series[-self.__num_forecast:]
        time_series_train = self.__time_series[:-self.__num_forecast]
        db_ts = DelayVectorDB(time_series_train, self.__m, self.__tau, delta=self.__num_forecast, dates=self.__dates)
        X, y = db_ts.get_vectors()
        nn = NearestNeighbors(epsilon=self.__eps)
        nn.fit(X, y)
        new_vector = last_vector(self.__time_series[:-self.__num_forecast], [self.__m, ], [self.__tau, ])
        forecast = nn.predict(new_vector)
        score = self.__error_fun(validation_set, forecast)
        if test:
            self.plotting(forecast, validation_set)
        return score

    def nss_mlp(self, test=False, hidden_layer_sizes=(100, 100), train=True, path_model=None):
        validation_set = self.__time_series[-self.__num_forecast:]
        time_series_train = self.__time_series[:-self.__num_forecast]
        db_ts = DelayVectorDB(time_series_train, self.__m, self.__tau, delta=self.__num_forecast, dates=self.__dates)
        X, y = db_ts.get_vectors()
        if train:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, early_stopping=True)
            mlp.fit(X, y)
        else:
            if path_model is not None:
                mlp = pickle.load(open(path_model, 'rb'))
        new_vector = last_vector(self.__time_series[:-self.__num_forecast], [self.__m, ], [self.__tau, ], dates=self.__dates)
        forecast = mlp.predict(new_vector.reshape(1, -1))
        # score = self.__error_fun(validation_set, forecast)
        if test:
            self.plotting(forecast.ravel(), validation_set)
        return 1#score

    @staticmethod
    def plotting(forecast, validation_set):
        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        plt.rcParams.update({'font.size': 20})
        plt.plot(forecast, marker='o', label='forecast')
        plt.plot(validation_set, marker='v', label='validation')
        plt.legend()
        plt.show()

    def one_hour_ahead_mlp(self, delta, h, test=False, hidden_layer_sizes=(100, 100, 100), train=True, path_model=None):
        validation_set = self.__time_series[-self.__num_forecast:]
        forecast = list()
        time_series_train = self.__time_series[:-self.__num_forecast]
        db_ts = DelayVectorDB(time_series_train, self.__m, self.__tau, delta=delta, h=h, dates=self.__dates)
        X, y = db_ts.get_vectors()
        mlp = None
        if train:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, early_stopping=True)
            mlp.fit(X, y)
        else:
            if path_model is not None:
                mlp = pickle.load(open(path_model, 'rb'))
        for i in range(self.__num_forecast, 0, -1):
            new_vector = last_vector(self.__time_series[:-i], [self.__m,], [self.__tau,], dates=self.__dates)
            forecast.append(mlp.predict(new_vector.reshape(1, -1))[0])
            # X = np.vstack((X, new_vector))
            # y = np.vstack((y, np.append(y[-1][1:], [validation_set[np.abs(i - self.__num_forecast)]])))
        # todo: test this part more
        # I'm not convinced of this calculation, I think the length of the forecast set should also include h
        forecast = np.array(forecast)[:len(validation_set) - (delta - 1) - 1]

        idx_val = np.array(range(delta))
        db_val = np.array([validation_set[idx_val + i] for i in range(len(validation_set) - (delta - 1))])[h - 1:]
        score = np.mean([self.__error_fun(db_val[:, i], forecast[:, i]) for i in range(delta)])
        if test:
            self.plotting(forecast[:, 0], db_val[:, 0])
        return score, forecast, db_val, mlp

    def one_day_ahead_mlp(self, delta, h, hidden_layer_sizes=(100, 100), test=False, train=True, path_model=None):
        validation_set = self.__time_series[-self.__num_forecast:]
        forecast = list()
        time_series_train = self.__time_series[:-self.__num_forecast]
        db_ts = DelayVectorDB(time_series_train, self.__m, self.__tau, delta=delta, h=h, dates=self.__dates)
        X, y = db_ts.get_vectors()
        mlp = None
        if train:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, early_stopping=True)
            mlp.fit(X, y)
        else:
            if path_model is not None:
                mlp = pickle.load(open(path_model, 'rb'))

        for i in range(self.__num_forecast, 0, -delta):
            nn = NearestNeighbors(epsilon=self.__eps)
            nn.fit(X, y)
            new_vector = last_vector(self.__time_series[:-i], [self.__m, ], [self.__tau, ], dates=self.__dates)
            forecast.append(mlp.predict(new_vector.reshape(1, -1))[0])
            for j in range(i, i - delta, -1):
                vector = last_vector(self.__time_series[:-j], [self.__m, ], [self.__tau, ], dates=self.__dates)
                X = np.vstack((X, vector))
                y = np.vstack((y, np.append(y[-1][1:], [validation_set[np.abs(j - self.__num_forecast)]])))
        forecast = np.array(forecast).ravel()
        score = self.__error_fun(validation_set, forecast)
        if test:
            self.plotting(forecast, validation_set)
        return score, forecast, validation_set, mlp

    def osa_mlp(self, hidden_layer_sizes=(100, 100), test=False, train=True, path_model=None):
        validation_set = self.__time_series[-self.__num_forecast:]
        forecast = list()
        time_series_train = self.__time_series[:-self.__num_forecast]
        db_ts = DelayVectorDB(time_series_train, self.__m, self.__tau, dates=self.__dates)
        X, y = db_ts.get_vectors()
        mlp = None
        if train:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, early_stopping=True)
            mlp.fit(X, y)
        else:
            if path_model is not None:
                mlp = pickle.load(open(path_model, 'rb'))

        for i in range(self.__num_forecast, 0, -1):
            new_vector = last_vector(self.__time_series[:-i], [self.__m,], [self.__tau,], dates=self.__dates)
            forecast.append(mlp.predict(new_vector.reshape(1, -1))[0])
            X = np.vstack((X, new_vector))
            y = np.vstack((y, [validation_set[np.abs(i - self.__num_forecast)]]))

        forecast = np.array(forecast)
        if test:
            self.plotting(forecast, validation_set)
        return self.__error_fun(validation_set, forecast)
