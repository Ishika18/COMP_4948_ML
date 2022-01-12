# SGDClassifier is an altered version of SGDRegresspr
# SGDClassifier is designed to make binary and multi-class predictions
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import SGDClassifier


def sgd_classification(X_trainScaled, X_testScaled, y_train, y_test):
    print("\nStochastic Gradient Descent")

    clf = SGDClassifier()
    clf.fit(X_trainScaled, y_train)

    y_pred = clf.predict(X_testScaled)

    # Show confusion matrix and accuracy scores.
    confusion_matrix = pd.crosstab(y_test, y_pred,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])

    print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix)
