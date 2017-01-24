import pandas as pd
from sklearn.metrics import mean_squared_error
from classes.ModelTypes import ModelTypes


class ModelEvaluator(object):
    @classmethod
    def evaluate(cls, meta_model, data):
        x_data = data.ix[:, data.columns.difference(pd.Index(["target", "healthCode"]))]
        y_data = pd.Series(data["target"].values, name="actual")
        prediction_data = pd.Series(meta_model.model.predict(x_data), name="prediction")

        if meta_model.name in ModelTypes.CATEGORICAL:
            return cls.evaluate_accuracy(y_data, prediction_data)
        elif meta_model.name in ModelTypes.CONTINUOUS:
            return cls.evaluate_mse(y_data, prediction_data)
            # todo - else throw execption

    @classmethod
    def evaluate_accuracy(cls, y_data, prediction_data):
        y_data = pd.Series(y_data, name="actual")
        prediction_data = pd.Series(prediction_data, name="prediction")
        results = pd.concat([y_data, prediction_data], axis=1)
        count_matched = 0
        for index, row in results.iterrows():
            if row.values[0] == row.values[1]:
                count_matched += 1

        return float(count_matched) / float(len(y_data))

