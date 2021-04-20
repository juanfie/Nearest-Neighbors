from NearestNeighborsModule.data_manipulation import data_read_dat
from NearestNeighborsModule.DelayVectorDB import DelayVectorDB
import pandas as pd

if __name__ == '__main__':
    #time_series = data_read_dat('/Users/rafa/TimeSeries/DataSets/synthetic/henon.dat')
    df = pd.read_csv('/Users/rafa/Dropbox/CENACE_forecasting/DEMANDA_NETA_SIN.csv', index_col='FECHA')
    time_series = df['DEMANDA_NETA_SIN'].values
    dates = df['DEMANDA_NETA_SIN'].index
    m = 10
    tau = 1
    num_forecast = 50
    delta = 4
    h = 2

    validation_set = time_series[-num_forecast:]
    forecast = list()
    time_series_train = time_series[:-num_forecast]
    db_ts = DelayVectorDB(time_series_train, m, tau, delta=delta, h=h, dates=dates)
    X, y = db_ts.get_vectors()
    print('the end')