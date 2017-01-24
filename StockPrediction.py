import pandas as pd

import CustomShapelets
from classes.ModelEvaluator import ModelEvaluator
from classes.ModelFactory import ModelFactory
from classes.ModelNames import ModelNames

try:
    from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
    # quotes_historical_yahoo_ochl was named quotes_historical_yahoo before matplotlib 1.4
    from matplotlib.finance import quotes_historical_yahoo as quotes_historical_yahoo_ochl
from matplotlib.collections import LineCollection
import datetime
import numpy as np
import dtw

# --------------------------------------------------------------------
symbol_dict = {
    'TOT': 'Total',
    'XOM': 'Exxon',
    'CVX': 'Chevron',
    'COP': 'ConocoPhillips',
    'VLO': 'Valero Energy',
    'MSFT': 'Microsoft',
    'IBM': 'IBM',
    'TWX': 'Time Warner',
    'CMCSA': 'Comcast',
    'CVC': 'Cablevision',
    'YHOO': 'Yahoo',
    'HPQ': 'HP',
    'AMZN': 'Amazon',
    'TM': 'Toyota'}


# 'CAJ': 'Canon',
# 'MTU': 'Mitsubishi',
# 'SNE': 'Sony',
# 'F': 'Ford',
# 'HMC': 'Honda',
# 'NAV': 'Navistar',
# 'NOC': 'Northrop Grumman',
# 'BA': 'Boeing',
# 'KO': 'Coca Cola',
# 'MMM': '3M',
# 'MCD': 'Mc Donalds',
# 'PEP': 'Pepsi',
# 'MDLZ': 'Kraft Foods',
# 'K': 'Kellogg',
# 'UN': 'Unilever',
# 'MAR': 'Marriott',
# 'PG': 'Procter Gamble',
# 'CL': 'Colgate-Palmolive',
# 'GE': 'General Electrics',
# 'WFC': 'Wells Fargo',
# 'JPM': 'JPMorgan Chase',
# 'AIG': 'AIG',
# 'AXP': 'American express',
# 'BAC': 'Bank of America',
# 'GS': 'Goldman Sachs',
# 'AAPL': 'Apple',
# 'SAP': 'SAP',
# 'CSCO': 'Cisco',
# 'TXN': 'Texas instruments',
# 'XRX': 'Xerox',
# 'LMT': 'Lookheed Martin',
# 'WMT': 'Wal-Mart',
# 'WBA': 'Walgreen',
# 'HD': 'Home Depot',
# 'GSK': 'GlaxoSmithKline',
# 'PFE': 'Pfizer',
# 'SNY': 'Sanofi-Aventis',
# 'NVS': 'Novartis',
# 'KMB': 'Kimberly-Clark',
# 'R': 'Ryder',
# 'GD': 'General Dynamics',
# 'RTN': 'Raytheon',
# 'CVS': 'CVS',
# 'CAT': 'Caterpillar',
# 'DD': 'DuPont de Nemours'}


# ------------------------------------------------------------------------------
def normalize(df):
    return np.array([(x - df.mean()) / (0.5 * (df.max() - df.min())) for x in df]).astype(np.float)


# ------------------------------------------------------------------------------


def retrieveStockData(symbol_dict, startDate, endDate):
    symbols, names = np.array(list(symbol_dict.items())).T
    quotes = [quotes_historical_yahoo_ochl(symbol, startDate, endDate, asobject=True)
              for symbol in symbols]
    data = np.array([q.close for q in quotes]).astype(np.float)
    data = np.array([normalize(x) for x in data]).astype(np.float)
    data = dict(zip(symbols, data))
    return data


import math as math


def generateClasses(arr):
    if arr[-1] > arr[-2]:
        return (arr[:-1], 1)
    return (arr[:-1], 0)


def create_raw_data(stock_data, L, SW):
    valid_index = [i for i in range(len(stock_data) - L) if i % SW == 0]
    sub_stock_data = []
    for i in valid_index:
        sub_train_set = stock_data[i:L + i]
        sub_stock_data.append(sub_train_set)

    labeled_data = [generateClasses(sub_stock_data[i]) for i in range(len(sub_stock_data))]
    return labeled_data


def create_avg_raw_data(raw_data, raw_data_neighbhours=[]):
    results = []
    for ts_idx, ts in enumerate(raw_data):
        ts_x, ts_y = ts
        arrays = [raw_data_neighbhour[ts_idx][0] for raw_data_neighbhour in raw_data_neighbhours]
        arrays.append(ts_x)
        arrays_mean = np.mean(arrays, axis=0)
        results.append((arrays_mean, ts_y))
    return results


# -------------------------------------
startDate = datetime.datetime(2014, 1, 1)
endDate = datetime.datetime(2015, 12, 30)
data = retrieveStockData(symbol_dict, startDate, endDate)

stock_data = data["MSFT"]
ts_len = 10
sw_len = 4
candidate_max_len = 5
candidate_min_len = 3

# ------- Create Raw Data --------------------
# -- raw data
raw_data = create_raw_data(stock_data, ts_len, sw_len)

# -- raw data avg
distance_matrix = dtw.calcualte_distance(data)
topN = dtw.get_top_n_similar_stocks("MSFT", 5, distance_matrix)
raw_data_neighbhours = [create_raw_data(data[key], ts_len, sw_len) for key in topN.keys()]
# for key in topN.keys():
#     x = create_raw_data(data[key], ts_len, sw_len)
#     raw_data_neighbhours.append(x)
avg_raw_data = create_avg_raw_data(raw_data, raw_data_neighbhours)

# train_size = int(round(len(raw_data) * 0.7))
# train_data = raw_data[:train_size]
# test_data = raw_data[train_size:len(raw_data)]
train_size = int(round(len(avg_raw_data) * 0.7))
train_data = avg_raw_data[:train_size]
test_data = avg_raw_data[train_size:len(avg_raw_data)]

candidates = CustomShapelets.generate_candidates(train_data, candidate_max_len, candidate_min_len)
x_train_data = CustomShapelets.extract_shapelet_features(train_data, candidates)
y_train_data = [nn[1] for nn in train_data]
model = ModelFactory.get_model(ModelNames.RANDOM_FOREST)
model_baseline = ModelFactory.get_model(ModelNames.CATEGORICAL_BASELINE)
model = model.fit(x_train_data, y_train_data)
model_baseline = model_baseline.fit(x_train_data, pd.Series(y_train_data))
x_test_data = CustomShapelets.extract_shapelet_features(test_data, candidates)
y_test_data = [nn[1] for nn in test_data]
prediction_test_data = model.predict(x_test_data)
baseline_prediction_test_data = model_baseline.predict(x_test_data)
accuracy = ModelEvaluator.evaluate_accuracy(y_test_data, prediction_test_data)
accuracy_baseline = ModelEvaluator.evaluate_accuracy(y_test_data, baseline_prediction_test_data)


print("foo")

print("Hi")
