import numpy as np
from sklearn.linear_model import Lasso
from tensorflow import metrics


def performLassorRegression(X_train, X_test, y_train, y_test, alpha):
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit(X_train, y_train)
    y_pred = lassoreg.predict(X_test)
    print("\n***Lasso Regression Coefficients ** alpha=" + str(alpha))
    print(lassoreg.intercept_)
    print(lassoreg.coef_)
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# USAGE
# alphaValues = [0, 0.1, 0.5, 1]
# for i in range(0, len(alphaValues)):
#     performLassorRegression(X_train, X_test, y_train, y_test,
#                      alphaValues[i])
