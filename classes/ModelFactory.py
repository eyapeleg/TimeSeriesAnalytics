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
            return DecisionTreeClassifier(random_state=0)
        elif model_name == ModelNames.RANDOM_FOREST:
            return RandomForestClassifier(n_estimators=100)
            ## elif todo - throw execpetion
