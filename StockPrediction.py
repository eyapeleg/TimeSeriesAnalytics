import CustomShapelets
import Modeling

try:
     from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
     # quotes_historical_yahoo_ochl was named quotes_historical_yahoo before matplotlib 1.4
    from matplotlib.finance import quotes_historical_yahoo as quotes_historical_yahoo_ochl
from matplotlib.collections import LineCollection
import datetime
import numpy as np
from collections import defaultdict
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
        'TM': 'Toyota',
        'CAJ': 'Canon',
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

startDate = datetime.datetime(2015, 1, 1)
endDate = datetime.datetime(2015, 12, 30)
# ------------------------------------------------------------------------------
def normalize(df):
    return np.array([(x-df.mean())/(0.5*(df.max()-df.min())) for x in df]).astype(np.float)

# ------------------------------------------------------------------------------


def retrieveStockData(symbol_dict, startDate, endDate):
    symbols, names = np.array(list(symbol_dict.items())).T
    quotes = [quotes_historical_yahoo_ochl(symbol, startDate, endDate, asobject=True)
              for symbol in symbols]
    data = np.array([q.close for q in quotes]).astype(np.float)
    data = np.array([normalize(x) for x in data]).astype(np.float)
    data = dict(zip(symbols,data))
    return data


import math as math

def generateClasses(arr):
    if arr[-1] > arr[-2]:
        return (arr[:-1], 1)
    return (arr[:-1], 0)


def createTrainingData(stock_data, L, SW):
    valid_index = [i for i in range(len(stock_data)-L) if i % SW == 0]
    sub_stock_data = []
    for i in valid_index:
        sub_train_set = stock_data[i:L+i]
        sub_stock_data.append(sub_train_set)

    labeled_data = [generateClasses(sub_stock_data[i]) for i in range(len(sub_stock_data))]
    return labeled_data
#-------------------------------------
data = retrieveStockData(symbol_dict,startDate,endDate)
# distance_matrix = dtw.calcualte_distance(data)
# topN = dtw.get_top_n_similar_stocks("MSFT", 5, distance_matrix)
sub_train_data = createTrainingData(data["MSFT"], 10, 4)
data_len = len(sub_train_data)
train_size = int(round(data_len*0.7))
train_set = sub_train_data[:train_size]
test_set = sub_train_data[train_size:data_len]
test_set = [x[0] for x in test_set]


features, shapelet_mapping = CustomShapelets.extract_shapelet_features(train_set)
labels = np.array([train_set[i][-1] for i in range(len(train_set))])
X = np.array(features.values())

import pandas as pd
# df_lables = pd.DataFrame(labels)
# df_features = pd.DataFrame(features)

model = Modeling.decision_tree(X, labels)
print 'a'


