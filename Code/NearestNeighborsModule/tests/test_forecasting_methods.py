from NearestNeighborsModule.forecasting_methods import ForecastingMethods
from NearestNeighborsModule.DelayVectorDB import convert_dates
import pandas as pd
from NearestNeighborsModule.metrics import mape, mae
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    file_path = '/Users/rafa/Dropbox/CENACE_forecasting/DEMANDA_NETA_SIN.csv'  # sys.argv[1]
    name_time_series = 'DEMANDA_ORI'  # sys.argv[2]

    df_all = pd.read_csv(file_path, index_col='FECHA')

    idx_sub_sampling = [i for i in range(0, df_all.__len__(), 15)]
    df = df_all[name_time_series][idx_sub_sampling]
    df = df[:df.index.get_loc('2018-06-25 00:00:00')]

    time_series_scaler = MinMaxScaler()
    scaled_time_series_values = time_series_scaler.fit_transform(df.values.reshape(-1, 1))
    scaled_time_series = pd.Series(scaled_time_series_values.ravel(), index=df.index)

    time_series = scaled_time_series.values
    dates = convert_dates(scaled_time_series.index)
    # DEMANDA_CEL [6, 4, 0.34100608]
    # DEMANDA_NES [8, 1, 0.14140621]
    # DEMANDA_NOR [2, 46, 0.05696183]
    # DEMANDA_NTE [2, 4, 0.05038196]
    # DEMANDA_OCC [13, 31, 0.34437311]
    # DEMANDA_ORI [2, 6, 0.06026154]
    # DEMANDA_PEN [1, 22, 0.02691927]

    (m, tau, eps) = [2, 1, 0.06026154]
    num_forecast = 96
    delta = 4
    h = 2

    fm = ForecastingMethods(time_series, m, tau, eps, num_forecast, mape, dates)
    score, forecast, db_val = fm.one_hour_ahead(delta, h, test=True)
    print(score)
    forecast_actual = time_series_scaler.inverse_transform(forecast)
    db_val_actual = time_series_scaler.inverse_transform(db_val)
    print(mae(forecast_actual[:, 0], db_val_actual[:, 0]))
    print(forecast_actual[:, 0] - db_val_actual[:, 0])