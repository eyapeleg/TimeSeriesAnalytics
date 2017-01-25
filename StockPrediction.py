import csv
import logging

import pandas as pd
import time
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
print("starting program...")

# ----------------- Functions   ---------------------------------------
def normalize(df):
    return np.array([(x - df.mean()) / (0.5 * (df.max() - df.min())) for x in df]).astype(np.float)

def retrieveStockData(symbol_dict, startDate, endDate):
    symbols, names = np.array(list(symbol_dict.items())).T
    quotes = [quotes_historical_yahoo_ochl(symbol, startDate, endDate, asobject=True)
              for symbol in symbols]
    data = np.array([q.close for q in quotes]).astype(np.float)
    data = np.array([normalize(x) for x in data]).astype(np.float)
    data = dict(zip(symbols, data))
    return data

def generateClasses(arr):
    if arr[-1] > arr[-2]:
        return (arr[:-1], 1)
    return (arr[:-1], 0)

def create_train_data(stock_data, L, SW):
    valid_index = [i for i in range(len(stock_data) - L) if i % SW == 0]
    sub_stock_data = []
    for i in valid_index:
        sub_train_set = stock_data[i:L + i]
        sub_stock_data.append(sub_train_set)

    labeled_data = [generateClasses(sub_stock_data[i]) for i in range(len(sub_stock_data))]
    return labeled_data

def create_avg_train_data(raw_data, raw_data_neighbhours):
    results = []

    if len(raw_data_neighbhours) == 0:
        return raw_data

    for ts_idx, ts in enumerate(raw_data):
        ts_x, ts_y = ts
        arrays = [raw_data_neighbhour[ts_idx][0] for raw_data_neighbhour in raw_data_neighbhours]
        arrays.append(ts_x)
        arrays_mean = np.mean(arrays, axis=0)
        results.append((arrays_mean, ts_y))
    return results


def split_train_test_set(data):
    train_size = int(round(len(data) * 0.7))
    train_data = data[:train_size]
    test_data = data[train_size:len(data)]
    return train_data, test_data

# -------------- Params ---------------------------------
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
    'TM': 'Toyota','CAJ': 'Canon',
    'MTU': 'Mitsubishi',
    'SNE': 'Sony',
    'F': 'Ford',
    'HMC': 'Honda',
    'NAV': 'Navistar',
    'NOC': 'Northrop Grumman',
    'BA': 'Boeing',
    'KO': 'Coca Cola',
    'MMM': '3M',
    'MCD': 'Mc Donalds',
    'PEP': 'Pepsi',
    'MDLZ': 'Kraft Foods',
    'K': 'Kellogg',
    'UN': 'Unilever',
    'MAR': 'Marriott',
    'PG': 'Procter Gamble',
    'CL': 'Colgate-Palmolive',
    'GE': 'General Electrics',
    'WFC': 'Wells Fargo',
    'JPM': 'JPMorgan Chase',
    'AIG': 'AIG',
    'AXP': 'American express',
    'BAC': 'Bank of America',
    'GS': 'Goldman Sachs',
    'AAPL': 'Apple',
    'SAP': 'SAP',
    'CSCO': 'Cisco',
    'TXN': 'Texas instruments',
    'XRX': 'Xerox',
    'LMT': 'Lookheed Martin',
    'WMT': 'Wal-Mart',
    'WBA': 'Walgreen',
    'HD': 'Home Depot',
    'GSK': 'GlaxoSmithKline',
    'PFE': 'Pfizer',
    'SNY': 'Sanofi-Aventis',
    'NVS': 'Novartis',
    'KMB': 'Kimberly-Clark',
    'R': 'Ryder',
    'GD': 'General Dynamics',
    'RTN': 'Raytheon',
    'CVS': 'CVS',
    'CAT': 'Caterpillar',
    'DD': 'DuPont de Nemours'}

sw_len = 4
candidate_max_len = 7
candidate_min_len = 5
l_arr = [10, 15, 20]
nn_arr = [0, 2, 4, 6, 8, 10]
classifier_names = [ModelNames.RANDOM_FOREST,
                    ModelNames.DT,
                    ModelNames.LOGISTIC_REGRESSION,
                    ModelNames.CATEGORICAL_BASELINE]
stocks_names = symbol_dict.keys()

# ------------ Retrieve All Stocks Data -------------------------
startDate = datetime.datetime(2014, 1, 1)
endDate = datetime.datetime(2015, 12, 30)

s = time.clock()
all_stocks_data = retrieveStockData(symbol_dict, startDate, endDate)
print 'retrieve stock data in {} secs'.format(time.clock() - s)

# ------- Create Distance Matrix --------------------
s = time.clock()
distance_matrix = dtw.calcualte_distance(all_stocks_data)
print 'calculate DTW distance matrix in {} secs'.format(time.clock() - s)

# ------- Create Models --------------------
with open('results.csv', 'wb') as fp:
    csv_writer = csv.writer(fp, delimiter=',')
    for stock_name in stocks_names:
        stock_data = all_stocks_data[stock_name]
        for nn_size in nn_arr:
            topN = dtw.get_top_n_similar_stocks(stock_name, nn_size, distance_matrix)
            for l in l_arr:
                raw_data_neighbhours = [create_train_data(all_stocks_data[key], l, sw_len) for key in topN.keys()]
                raw_data = create_train_data(stock_data, l, sw_len)
                avg_raw_data = create_avg_train_data(raw_data, raw_data_neighbhours)

                train_data, test_data = split_train_test_set(avg_raw_data)

                candidates = CustomShapelets.generate_candidates(train_data, candidate_max_len, candidate_min_len)

                x_train_data = CustomShapelets.extract_shapelet_features(train_data, candidates)
                y_train_data = [nn_train[1] for nn_train in train_data]

                x_test_data = CustomShapelets.extract_shapelet_features(test_data, candidates)
                y_test_data = [nn_test[1] for nn_test in test_data]

                for classifier_name in classifier_names:
                    classifier = ModelFactory.get_model(classifier_name)
                    classifier = classifier.fit(x_train_data, y_train_data)

                    prediction_train_data = classifier.predict(x_train_data)
                    accuracy_train = ModelEvaluator.evaluate_accuracy(y_train_data, prediction_train_data)

                    prediction_test_data = classifier.predict(x_test_data)
                    accuracy_test = ModelEvaluator.evaluate_accuracy(y_test_data, prediction_test_data)

                    print 'stock_name: {}, nn: {}, length: {}, classifier: {}, accuracy train: {}, ' \
                          'accuracy test: {}'.format(stock_name, nn_size, l, classifier_name,
                                                     accuracy_train, accuracy_test)
                    result_item = [stock_name, nn_size, l, classifier_name, accuracy_train, accuracy_test]
                    csv_writer.writerow(result_item)
