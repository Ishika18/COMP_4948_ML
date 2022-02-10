import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import SGDRegressor
from global_constants import PATH

CSV_DATA = "winequality.csv"
dataset = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,  # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',',
                      names=('fixed acidity', 'volatile acidity', 'citric acid',
                             'residual sugar', 'chlorides', 'free sulfur dioxide',
                             'total sulfur dioxide', 'density', 'pH', 'sulphates',
                             'alcohol', 'quality'))

X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates',
             'alcohol']]

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X_withConst = sm.add_constant(X)
y = dataset['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X_withConst, y,
                                                    test_size=0.2, random_state=0)


def performLinearRegression(X_train, X_test, y_train, y_test):
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions


predictions = performLinearRegression(X_train, X_test, y_train, y_test)


def performSGD(X_train, X_test, y_train, y_test, scalerY):
    sgd = SGDRegressor(verbose=1)
    sgd.fit(X_train, y_train)
    print("\n***SGD=")
    predictions = sgd.predict(X_test)
    # print(predictions)

    y_test_unscaled = scalerY.inverse_transform(y_test)
    predictions_unscaled = scalerY.inverse_transform(predictions.reshape(-1, 1))
    # print(predictions_unscaled)

    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test_unscaled,
                                             predictions_unscaled)))


from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler()
scalerX.fit(X)
x2Scaled = scalerX.transform(X)

scalerY = MinMaxScaler()
reshapedY = y.reshape(-1, 1)
scalerY.fit(reshapedY)
yScaled = scalerY.transform(reshapedY)
X_train, X_test, y_train, y_test = train_test_split(x2Scaled, yScaled,
                                                    test_size=0.2, random_state=0)
performSGD(X_train, X_test, y_train, y_test, scalerY)
