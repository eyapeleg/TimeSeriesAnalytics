import pandas as pd


class ModelTrainer(object):

    @staticmethod
    def fit(model, data):
        x_data = data.ix[:, data.columns.difference(pd.Index(["target"]))]
        y_data = data["target"]
        return model.fit(x_data, y_data)
