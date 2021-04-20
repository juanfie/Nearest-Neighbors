import numpy as np
import datetime


class DelayVectorDB(object):
    def __init__(self, time_series, m, tau=1, delta=1, h=1, i=1, dates=None, meta_vars=False):
        self.__time_series = time_series
        self.__m = [m, ]
        self.__tau = [tau, ]
        self.__delta = delta
        self.__h = h
        self.__i = i
        self.__dates = dates
        self.__meta_vars = meta_vars
        self.__db_dv, self.__db_nv = self.form_seasonal_db()

    def form_seasonal_db(self):
        """Creates a Database (a.k.a. design matrix) from a Time Series.

        Named arguments:
        Array -- a numpy array containing the Time Series
                 if Array is a list, it is converted to numpy array
        m     -- a list of dimensions
        tau   -- a list of subsamplings
        delta -- the number of generated future values
        h     -- how far appart the future values are

        Returns:
        Two numpy arrays containing the Training Set Database.
        dbDV is an array of Delay vectors
        dbNV is an array of targets (future values)"""
        sdvi = self.seasonal_dv_index()
        now = sdvi[-1]
        fi = self.form_future_index(now)
        ind_dv = [sdvi + i for i in range(len(self.__time_series) - fi[-1])]
        ind_nv = [fi + i for i in range(len(self.__time_series) - fi[-1])]

        db_dv = np.array([self.__time_series[r] for r in ind_dv])
        if self.__dates is not None:
            db_dv = np.hstack((self.__dates[:len(db_dv)], db_dv))

        db_nv = np.array([self.__time_series[r] for r in ind_nv])
        return db_dv, db_nv

    def seasonal_dv_index(self):
        """Forms a SeasonalDelay Vector (DV) index
        Named arguments:
        m   -- a list of dimensions
        tau -- a list of subsamplings
        m and tau must be of equal lengths.
        Each correspondig pair (m[i], tau[i]) defines a seasonal pattern
        Returns:
        A numpy array containing the DV indices."""
        res = np.array(list(map(lambda x, y: self.dv_index(x, y), self.__m, self.__tau)))
        largest = res[-1, -1]
        res = np.array(list(map(lambda x: x - x[-1] + largest, res)))
        res = res.flatten()
        res = np.unique(res)
        return res

    def form_future_index(self, now=0):
        """Forms indices of the following (future) values of a TS.
        Those indices correspond to the places where forecasts are required.
        Named arguments:
        now   -- the present time
        delta -- the number of generated future values
        h     -- how far appart the future values are
        Returns:
        A numpy array of future indices."""
        return np.array(range(now + self.__h, (now + self.__h) + self.__delta * self.__i, self.__i))

    @staticmethod
    def dv_index(m=3, tau=2):
        """Forms a Delay Vector (DV) index
        Named arguments:
        m   -- dimension
        tau -- sampling distance
        Returns:
        A numpy array containing the DV index."""
        return np.array(list(range(0, m * tau, tau)))

    def get_vectors(self):
        return self.__db_dv, self.__db_nv


def convert_dates(dates, features=1):
    """
    this method converts a list of string timestamps into datetime objects and then converts them into features
    :param dates: a list of timestamp strings
    :param features: a numpy array of date and time features
    :return:
    """
    temp = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dates]
    temp = extract_features(temp, features)
    return np.array(temp)


def extract_features(dates, features):
    """
    Converts a list datetime objects into temporal features
    :param dates: list of datetime objects
    :param features: flag to determine which temporal features to extract
    :return: list of lists with temporal features
    """
    time_converted = None
    if features == 1:
        time_converted = [[(x.weekday() / 7),
                           ((x.hour * 60 + x.minute) / 1440)] for x in dates]
    elif features == 2:
        time_converted = [[(x.month / 12),
                           (x.weekday() / 7),
                           ((x.hour * 60 + x.minute) / 1440)] for x in dates]
    elif features == 3:
        time_converted = [[(convert_to_quarter(x.month) / 4),
                           (x.weekday() / 7),
                           ((x.hour * 60 + x.minute) / 1440)] for x in dates]
    elif features == 4:
        time_converted = [[(convert_to_quarter(x.month) / 4),
                           (x.month / 12),
                           (x.weekday() / 7),
                           ((x.hour * 60 + x.minute) / 1440)] for x in dates]
    elif features == 5:
        time_converted = [[(convert_to_third(x.month) / 4),
                           (x.weekday() / 7),
                           ((x.hour * 60 + x.minute) / 1440)] for x in dates]
    elif features == 6:
        time_converted = [[(convert_to_third(x.month) / 4),
                           (x.month / 12),
                           (x.weekday() / 7),
                           ((x.hour * 60 + x.minute) / 1440)] for x in dates]
    return time_converted


def convert_to_quarter(month):
    if month in [1, 2, 3]:
        return 0
    elif month in [4, 5, 6]:
        return 1
    elif month in [7, 8, 9]:
        return 2
    elif month in [10, 11, 12]:
        return 3


def convert_to_third(month):
    if month in [1, 2, 3, 4]:
        return 0
    elif month in [5, 6, 7, 8]:
        return 1
    elif month in [9, 10, 11, 12]:
        return 2


def get_season(now):
    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [(0, (datetime.date(Y, 1, 1), datetime.date(Y, 3, 20))),
               (1, (datetime.date(Y, 3, 21), datetime.date(Y, 6, 20))),
               (2, (datetime.date(Y, 6, 21), datetime.date(Y, 9, 22))),
               (3, (datetime.date(Y, 9, 23), datetime.date(Y, 12, 20))),
               (0, (datetime.date(Y, 12, 21), datetime.date(Y, 12, 31)))]
    if isinstance(now, datetime.datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)