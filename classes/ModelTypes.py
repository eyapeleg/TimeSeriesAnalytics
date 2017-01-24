from ModelNames import ModelNames


class ModelTypes(object):
    CONTINUOUS = [ModelNames.GBT, ModelNames.LASSO, ModelNames.CONTINUOUS_BASELINE]
    CATEGORICAL = [ModelNames.CATEGORICAL_BASELINE, ModelNames.DT, ModelNames.RANDOM_FOREST]

    @classmethod
    def foo(cls):
        print "foo"

    def bar(self):
        self.foo()
