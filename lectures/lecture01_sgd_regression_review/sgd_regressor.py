import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Stochastic gradient descent models are sensitive to differences
# in scale so a StandardScaler is usually used.
from sklearn.preprocessing import StandardScaler


def sgd_regression(X_train, X_test, y_train, y_test):
    ###########################################################
    print("\nStochastic Gradient Descent")

    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)

    # SkLearn SGD classifier
    sgd = SGDRegressor(verbose=1)
    sgd.fit(X_trainScaled, y_train)
    predictions = sgd.predict(X_testScaled)
    print('Root Mean Squared Error:',
          np.sqrt(mean_squared_error(y_test, predictions)))
