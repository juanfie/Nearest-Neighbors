from NearestNeighborsModule.forecasting_methods import ForecastingMethods
from NearestNeighborsModule.DelayVectorDB import convert_dates
import pandas as pd
from NearestNeighborsModule.metrics import mape, mae
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    file_path = '/Users/rafa/Dropbox/CENACE_forecasting/DEMANDA_NETA_SIN.csv'  # sys.argv[1]

    df_all = pd.read_csv(file_path, index_col='FECHA')

    idx_sub_sampling = [i for i in range(0, df_all.__len__(), 15)]

    dict_params = {'DEMANDA_CEL': [6, 4, 0.34100608],
                   'DEMANDA_NES': [8, 1, 0.14140621],
                   'DEMANDA_NOR': [2, 46, 0.05696183],
                   'DEMANDA_NTE': [2, 4, 0.05038196],
                   'DEMANDA_OCC': [13, 31, 0.34437311],
                   'DEMANDA_ORI': [2, 6, 0.06026154],
                   'DEMANDA_PEN': [1, 22, 0.02691927]}

    demanda_neta_sin_forecast = np.zeros(92)
    demanda_neta_sin_validation = np.zeros(92)
    for name_time_series in ['DEMANDA_CEL', 'DEMANDA_NES', 'DEMANDA_NOR', 'DEMANDA_NTE', 'DEMANDA_OCC', 'DEMANDA_ORI', 'DEMANDA_PEN']:
        df = df_all[name_time_series][idx_sub_sampling]
        df = df[:df.index.get_loc('2018-06-25 00:00:00')]
        time_series_scaler = MinMaxScaler()
        scaled_time_series_values = time_series_scaler.fit_transform(df.values.reshape(-1, 1))
        scaled_time_series = pd.Series(scaled_time_series_values.ravel(), index=df.index)
        time_series = scaled_time_series.values
        dates = convert_dates(scaled_time_series.index)

        (m, tau, eps) = dict_params[name_time_series]
        num_forecast = 96
        delta = 4
        h = 2

        fm = ForecastingMethods(time_series, m, tau, eps, num_forecast, mape, dates)
        score, forecast, db_val = fm.one_hour_ahead(delta, h, test=False)
        # print(score)
        forecast_actual = time_series_scaler.inverse_transform(forecast)
        db_val_actual = time_series_scaler.inverse_transform(db_val)
        print('{}: {}'.format(name_time_series, mae(forecast_actual[:, 0], db_val_actual[:, 0])))
        # print(forecast_actual[:, 0] - db_val_actual[:, 0])

        demanda_neta_sin_forecast += forecast_actual[:, 0]
        demanda_neta_sin_validation += db_val_actual[:, 0]
    print(mae(demanda_neta_sin_forecast, demanda_neta_sin_validation))
    print(demanda_neta_sin_forecast - demanda_neta_sin_validation)
    print(demanda_neta_sin_forecast)
    plt.plot(demanda_neta_sin_forecast, marker='o', label='forecast')
    plt.plot(demanda_neta_sin_validation, marker='v', label='validation')
    plt.show()

