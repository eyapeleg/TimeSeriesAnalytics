import numpy as np
import pandas as pd


class CategoricalBaselineModel:
    def __init__(self):
        self.y_mode = None

    def fit(self, x_data, y_train_data):
        self.y_mode = pd.Series(y_train_data).value_counts().idxmax()
        return self

    def predict(self, x_data):
        baseline_prediction = np.empty(len(x_data))
        baseline_prediction.fill(self.y_mode)
        baseline_prediction = pd.Series(baseline_prediction)
        return baseline_prediction
