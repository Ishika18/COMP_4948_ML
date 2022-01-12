import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm
from sklearn.linear_model import  Ridge
from global_constants import PATH
from sklearn.preprocessing import MinMaxScaler

CSV_DATA = "winequality.csv"

dataset  = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,       # Don't include header row as part of data.
                      encoding = "ISO-8859-1", sep=',',
                      names=('fixed acidity', 'volatile acidity', 'citric acid',
                             'residual sugar', 'chlorides', 'free sulfur dioxide',
                             'total sulfur dioxide', 'density', 'pH', 'sulphates',
                             'alcohol', 'quality'))
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(dataset.head())
print(dataset.describe())

# Include only statistically significant columns.
X = dataset[['volatile acidity',
             'chlorides', 'total sulfur dioxide',
             'pH', 'sulphates','alcohol']]
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Stochastic gradient descent models are sensitive to differences
# in scale so a MinMax is usually used.
scalerX = MinMaxScaler()
scalerX.fit(X_train)

# Build scaler for y.
scalerY = MinMaxScaler()
reshapedYtrain = np.array(y_train).reshape(-1,1)
scalerY.fit(reshapedYtrain)

# Scale X_train, X_test and y_train.
X_trainScaled = scalerX.transform(X_train)
X_testScaled  = scalerX.transform(X_test)
y_trainScaled = scalerY.transform(reshapedYtrain)

# Add constant to scaled data.
X_trainScaled = sm.add_constant(X_trainScaled)
X_testScaled  = sm.add_constant(X_testScaled)

#---------------------------------------------------------------
# Perform OLS regression.
model       = sm.OLS(y_trainScaled, X_trainScaled).fit()
predictions = model.predict(X_testScaled) # make the predictions by the model
print(model.summary())

# Convert predictions to unscaled predicitons and compare with y_test.
unscaledPredictionsOLS = scalerY.inverse_transform(predictions.reshape(-1,1))
print('Root Mean Squared Error:',
      np.sqrt(mean_squared_error(y_test, unscaledPredictionsOLS)))

#---------------------------------------------------------------
# Perform Ridge regression.
print("\nRidge Regression")
ridge_reg   = Ridge(solver='auto')
ridge_reg.fit(X_trainScaled, y_trainScaled)
predictions = ridge_reg.predict(X_testScaled)

# Convert predictions to unscaled predicitons and compare with y_test.
unscaledPredictionsRidge = scalerY.inverse_transform(predictions.reshape(-1,1))
print('Root Mean Squared Error:',
      np.sqrt(mean_squared_error(y_test, unscaledPredictionsRidge)))
