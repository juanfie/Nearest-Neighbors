from NearestNeighborsModule.nnde import run
from NearestNeighborsModule.metrics import smape
from NearestNeighborsModule.DelayVectorDB import convert_dates
from NearestNeighborsModule.data_manipulation import data_read_dat
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys


if __name__ == '__main__':
    file_path = '/Users/rafa/Dropbox/Solar Forecasting/Data/reconstructedRadTrim_hourly.dat'  # sys.argv[1]
    time_series = data_read_dat(file_path)

    delta = 24
    h = 1
    num_forecast = 24 * 7
    best = run(time_series, num_forecast, smape, 'oda', delta, h, dates=None, num_jobs=2, maxiter=5, popsize=5)
