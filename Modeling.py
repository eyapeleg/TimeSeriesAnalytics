import statsmodels.api as sm
import pandas as pd
import subprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

7
def LogisticRegression(X, y):
    logit = sm.Logit(y, X)
    return logit.fit()

def decision_tree(X,y):
    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    return dt.fit(X, y)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")