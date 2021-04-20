from NearestNeighborsModule.forecasting_methods import ForecastingMethods
from NearestNeighborsModule.data_manipulation import data_read_dat
from NearestNeighborsModule.metrics import mape
import sys

if __name__ == '__main__':

    # [17, 1, 1.892]
    # [12, 29, 14.125]
    # [72, 9, 16.388]
    # [12, 15, 21.643]
    # [37, 9, 12.092]
    # [74, 20, 45.856]
    params_henon_nss = [[10, 47, 3.581], [11, 47, 3.857], [5, 48, 1.307],
                        [9, 49, 3.344], [11, 4, 4.227], [36, 13, 5.295]]

    params_mackey_nss = [[3, 50, 0.878], [4, 32, 1.063], [3, 50, 1.005],
                         [4, 28, 0.724], [4, 30, 1.252], [4, 35, 1.332]]

    params_lorenz_nss = [[2, 44, 31.773], [2, 44, 33.231], [2, 41, 32.523],
                         [2, 22, 37.628], [90, 6, 24.132], [2, 50, 47.014]]

    params_henon = [[4, 1, 0.061], [8, 1, 0.386], [7, 1, 0.416],
                    [14, 1, 3.222], [11, 1, 2.881], [2, 2, 1.379]]

    params_lorenz = [[47, 1, 3.138], [41, 3, 4.482], [86, 1, 15.696],
                     [86, 5, 30.855], [11, 50, 28.131], [44, 12, 23.694]]

    params_mackey = [[51, 23, 0.033], [19, 39, 0.320], [30, 34, 0.452],
                     [62, 18, 0.430], [19, 24, 0.794], [14, 12, 0.796]]

    params_rossler = [[17, 1, 1.892], [12, 29, 14.125], [72, 9, 16.388],
                      [12, 15, 21.643], [37, 9, 12.092], [74, 20, 45.856]]

    names = ['henon', 'lorenz', 'mackey', 'rossler']

    all_params = [params_henon, params_lorenz, params_mackey, params_rossler]
    file_path_normal = '/Users/rafa/TimeSeries/DataSets/Synthetic/{}.dat'  # sys.argv[1]
    file_path_noisy = '/Users/rafa/TimeSeries/DataSets/Synthetic/{}_noisy_{}.dat'

    for name, params in zip(names, all_params):
    #name = 'lorenz'
    #params = params_lorenz
        for i, p in zip(range(0, 30, 5), params):
            file_path = None
            if i == 0:
                file_path = file_path_normal.format(name)
            else:
                file_path = file_path_noisy.format(name, i)
            time_series = data_read_dat(file_path)
            (m, tau, eps) = p
            fm = ForecastingMethods(time_series, m, tau, eps, 50, mape)
            score = fm.osa(True)
            print(score)
    # print(forecast)
    # print(validation_set)


