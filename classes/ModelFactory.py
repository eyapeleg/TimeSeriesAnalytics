from sklearn.linear_model import LogisticRegression

from CategoricalBaselineModel import CategoricalBaselineModel
from ModelNames import ModelNames
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class ModelFactory(object):
    @staticmethod
    def get_model(model_name, model_params=None):
        if model_name == ModelNames.CATEGORICAL_BASELINE:
            return CategoricalBaselineModel()
        elif model_name == ModelNames.DT:
            return DecisionTreeClassifier(
                **{"min_samples_split": 10, "min_samples_leaf": 5, "max_depth": 3, "random_state": 0})
        elif model_name == ModelNames.RANDOM_FOREST:
            return RandomForestClassifier(
                **{"min_samples_split": 10, "min_samples_leaf": 5, "max_depth": 2, "n_jobs": -1, "n_estimators": 300, "max_features": 6})
        elif model_name == ModelNames.LOGISTIC_REGRESSION:
            return LogisticRegression()
        ## elif todo - throw execpetion
