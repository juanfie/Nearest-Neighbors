from NearestNeighborsModule.nnde import run
from NearestNeighborsModule.metrics import mape
from NearestNeighborsModule.DelayVectorDB import convert_dates
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys


if __name__ == '__main__':
    file_path = '/Users/rafa/Dropbox/CENACE_forecasting/DEMANDA_NETA_SIN.csv'#sys.argv[1]
    name_time_series = 'DEMANDA_NETA_SIN'#sys.argv[2]

    df_all = pd.read_csv(file_path, index_col='FECHA')

    idx_sub_sampling = [i for i in range(0, df_all.__len__(), 15)]
    df = df_all[name_time_series][idx_sub_sampling]
    df = df[:df.index.get_loc('2018-06-24 00:00:00')]

    time_series_scaler = MinMaxScaler()
    scaled_time_series_values = time_series_scaler.fit_transform(df.values.reshape(-1, 1))
    scaled_time_series = pd.Series(scaled_time_series_values.ravel(), index=df.index)

    time_series = scaled_time_series.values
    dates = convert_dates(scaled_time_series.index)

    delta = 4
    h = 2
    num_forecast = 96
    best = run(time_series, num_forecast, mape, 'oha', delta, h, dates=dates, num_jobs=2, maxiter=5, popsize=5)
