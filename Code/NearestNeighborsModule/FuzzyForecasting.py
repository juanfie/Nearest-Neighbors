import numpy as np
import skfuzzy as fuzz
from scipy.stats import norm
from NearestNeighborsModule.cython import distances
import matplotlib.pyplot as plt


class FuzzyForecasting(object):
    def __init__(self, num_fuzzy_sets, variable_fuzzy_sets=False, debug=False):
        self.__num_fuzzy_sets = num_fuzzy_sets

        self.__db_dv = None
        self.__db_nv = None

        self.__use_naive = True
        self.__naive_firings = 0

        self.FSs = None
        self.Rules = {}
        self.FiringStrengths = {}

        self.__variable_fuzzy_sets = variable_fuzzy_sets
        self.__range_values = None
        self.__time_series = None

    def fit(self, X, y, time_series):
        self.__db_dv = X
        self.__db_nv = y
        self.__time_series = time_series
        self.ts2fis()

    def ts2fis(self):
        """Creates the fuzzy sets and compiles the rules of the Fuzzy Inference System"""
        minx = np.min(self.__db_dv)
        maxx = np.max(self.__db_dv)
        if self.__variable_fuzzy_sets:
            self.FSs = self.create_variable_fuzzy_sets(minx, maxx)
        else:
            self.FSs = self.create_fuzzy_sets(minx, maxx)
        self.db2aa()
        # self.clean_rules()

    def create_fuzzy_sets(self, minx, maxx):
        """Calculates the range of the dependent variable and generates the linguistic terms"""
        output = []
        width = (maxx - minx) / float(self.__num_fuzzy_sets - 1)
        self.__range_values = np.arange(minx, maxx, 0.01)
        output.append(fuzz.trimf(self.__range_values, [minx, minx, minx + width]))
        space = np.linspace(minx, maxx, self.__num_fuzzy_sets)[1:-1]
        for p in space:
            output.append(fuzz.trimf(self.__range_values, [p - width, p, p + width]))
        output.append(fuzz.trimf(self.__range_values, [maxx - width, maxx, maxx]))
        return np.array(output)

    def create_variable_fuzzy_sets(self, minx, maxx):
        output = []
        ts_mean = np.mean(self.__time_series)
        ts_std_deviation = np.std(self.__time_series)
        self.__range_values = np.arange(minx, maxx, 0.01)
        mincum = norm(loc=ts_mean, scale=ts_std_deviation).cdf(minx)
        rangecum = 1 - mincum
        limits = [norm.ppf(norm(loc=ts_mean, scale=ts_std_deviation).cdf(x)) for x in
                  np.arange(mincum, 1, rangecum / self.__num_fuzzy_sets)]
        limits[0] = minx
        limits[-1] = maxx
        output.append(fuzz.trimf(self.__range_values, [limits[0], limits[0], limits[1]]))
        for i in range(1, self.__num_fuzzy_sets - 1):
            output.append(fuzz.trimf(self.__range_values, [limits[i - 1], limits[i], limits[i + 1]]))
        output.append(fuzz.trimf(self.__range_values, [limits[-2], limits[-1], limits[-1]]))
        return np.array(output)

    def db2aa(self):
        """Prepares the time series database to compile the activation rules"""
        db = np.copy(self.__db_dv)
        nv = np.copy(self.__db_dv)
        sdb = np.concatenate((db, nv), axis=1)
        minx = np.min(self.__db_dv)
        for v in sdb:
            # ants, cons, fstrs, d = self.extract_rule(v, minx)
            ants, cons, fstrs = self.extract_rule(v, minx)
            self.insert_rule(ants, cons)
            # self.insert_firing_strengths(ants, cons, fstrs, d)

    def insert_rule(self, ants, cons):
        if str(ants) in self.Rules:
            values = self.Rules[str(ants)]
            values = values + cons
            self.Rules[str(ants)] = list(set(values))
        else:
            self.Rules[str(ants)] = cons

    def insert_firing_strengths(self, ants, cons, fstrs, d):
        if str(ants) + str(cons) in self.FiringStrengths:
            values = self.FiringStrengths[str(ants) + str(cons)] + [(fstrs, d)]
            self.FiringStrengths[str(ants) + str(cons)] = values
        else:
            self.FiringStrengths[str(ants) + str(cons)] = [(fstrs, d)]

    def extract_rule(self, v, minx):
        """Extracts the activation rule for a given delay vector"""
        # b2list, fslist, d = self.belongs_to(v, minx)
        b2list, fslist = self.belongs_to(v, minx)
        ants, cons = self.create_fuzzy_rule(b2list)
        return ants, cons, fslist  # , d

    def belongs_to(self, listn, minx):
        allres = distances.membership(listn, self.__db_dv.shape[1] + 1,
                                      self.__num_fuzzy_sets, self.__range_values, self.FSs, minx, 0.01)
        # allres = self.membership(listn)
        # ds = self.obtain_d(allres)
        return [np.argmax(a) for a in allres], np.min([np.max(a) for a in allres])  # , np.mean(ds)

    def obtain_d(self, allres):
        X = [np.max(a) for a in allres]
        lts = [np.argmax(a) for a in allres]
        A = []
        B = []
        C = []
        for lt in lts:
            values = self.FSs[lt]
            a = 0
            c = 0
            for i in range(len(values)):
                if values[i] > 0:
                    a = self.__range_values[i]
                    A.append(self.__range_values[i])
                    break
            for i in range(len(values) - 1, -1, -1):
                if values[i] > 0:
                    c = self.__range_values[i]
                    C.append(self.__range_values[i])
                    break
            B.append((c - a) / float(2))
        return [(x - b) / float(b - a) if x < b else (x - b) / float(c - b) for x, a, b, c in zip(X, A, B, C)]

    def membership(self, listn):
        m = []
        for val in listn:
            m.append([fuzz.interp_membership(self.__range_values, lt, val) for lt in self.FSs])
        return np.array(m)

    def create_fuzzy_rule(self, b2list):
        ants = b2list[:self.__db_dv.shape[1]]
        cons = b2list[self.__db_dv.shape[1]:]
        return ants, cons

    def clean_rules(self):
        for key, value in self.Rules.items():
            self.Rules[key] = list(set(value))

    def predict(self, eval_vector):
        """Operation to calculate the forecast for a given evaluation vector"""
        minx = np.min(self.__db_dv)
        #b2list, fstrs, d = self.belongs_to(eval_vector, minx)
        b2list, fstrs = self.belongs_to(eval_vector, minx)
        ants = b2list[:self.__db_dv.shape[1]]
        all_cons = None
        try:
            all_cons = self.Rules[str(ants)]
            # if self.test_num_rules:
            #     self.num_rules_activated.append(len(all_cons))
        except KeyError:  # the antecedents do not exist in the database
            # all_cons = self.extract_close_rules(ants)
            # if self.test_num_rules:
            #     self.num_rules_activated.append(0)
            pass
        return self.calculate_forecast(all_cons, ants, eval_vector)

    def calculate_forecast(self, cons, ants, eval_vector):
        f = None
        temp_cons = list(cons)
        # if len(temp_cons) > 0:
        #     strengths = []
        #     defuzzyfied_values = []
        #     dist = []
        #     for a in ants:
        #         for c in temp_cons:
        #             try:
        #                 strengths.append(self.FiringStrengths[str(a) + str(c)][0][0])
        #                 dist.append(self.FiringStrengths[str(a) + str(c)][0][1])
        #                 defuzzyfied_values.append(fuzz.defuzz(self.range_values, self.FSs[c], 'centroid'))
        #             except KeyError:
        #                 defuzzyfied_values.append(fuzz.defuzz(self.range_values, self.FSs[c], 'centroid'))
        #                 dist.append(0)
        #                 strengths.append(1)
        #     f = np.sum((np.array(defuzzyfied_values) * np.array(strengths)) + np.array(dist)) / np.sum(strengths)
        if len(temp_cons) > 0:
            active_rules = self.select_active_rules(temp_cons)
            f = fuzz.defuzz(self.__range_values, active_rules, 'centroid')
        else:
            # self.naive_firings += 1
            f = eval_vector[-1]
        return f

    def select_active_rules(self, idx):
        if len(idx) == 1:
            return self.FSs[idx.pop()]
        else:
            lt = self.FSs[idx.pop()]
            return np.fmax(lt, self.select_active_rules(idx))

    @staticmethod
    def clean_cons(cons):
        return [int(x.replace('[', '').replace(']', '').replace('.', '')) for x in cons.split(',')]

    @staticmethod
    def strength(ants_val, cons_val):
        return np.prod(ants_val) * np.prod(cons_val)

    def debug_sets(self, output):
        for o in output:
            plt.plot(self.__range_values, o)
        plt.show()