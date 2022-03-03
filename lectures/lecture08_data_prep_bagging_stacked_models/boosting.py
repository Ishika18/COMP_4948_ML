from sklearn.model_selection import cross_val_score
from mlxtend.classifier      import EnsembleVoteClassifier
from xgboost                 import XGBClassifier, plot_importance
from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import datasets
import pandas as pd

ada_boost   = AdaBoostClassifier()
grad_boost  = GradientBoostingClassifier()
xgb_boost   = XGBClassifier()
boost_array = [ada_boost, grad_boost, xgb_boost]
eclf        = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost,
                                           xgb_boost], voting='hard')

labels = ['Ada Boost', 'Grad Boost', 'XG Boost', 'Ensemble']


iris = datasets.load_iris()
data = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})
X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']

for clf, label in zip([ada_boost, grad_boost, xgb_boost, eclf], labels):
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print("Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(),
                                                           scores.std(), label))
