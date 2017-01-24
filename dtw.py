import math

# TODO - move distace_matrix as a private field

def DTWDistance(s1, s2, w=7):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def calcualte_distance(data):
    distance_matrix = {}
    for i_k, i_v in data.iteritems():
        distance_matrix[i_k] = {}
        for i_ix, j in data.iteritems():
            distance = DTWDistance(i, j)
            distance_matrix[i][j] = distance

from collections import defaultdict

def init_distance_matrix(data):
    distance_matrix = {}
    for i_index, i in enumerate(data.keys()):
        distance_matrix[i] = {}
        for j_index, j in enumerate(data.keys()):
            if i == j:
               continue
            distance_matrix[i][j] = float("inf")
    return distance_matrix



def calcualte_distance(data):
    distance_matrix = init_distance_matrix(data)
    #distance_matrix = defaultdict(lambda: 0, distance_matrix)
    for i_index, i in enumerate(data.keys()):
        for j_index, j in enumerate(data.keys()):
            if j_index <= i_index:
                continue
            distance = DTWDistance(data[i], data[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

def get_top_n_similar_stocks(stock_name, n, distance_matrix):
    import operator
    candidate_stocks = distance_matrix[stock_name]
    sorted_stocks = sorted(candidate_stocks.items(), key=operator.itemgetter(1))
    if n > len(candidate_stocks):
        raise("lech ala kush shela ima shelcha!")
    return dict(sorted_stocks[:n])
