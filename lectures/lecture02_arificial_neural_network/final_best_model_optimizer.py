import pandas as pd
import numpy  as np
from sklearn                 import metrics
from sklearn.model_selection import train_test_split

PATH     = "/Users/pm/Desktop/DayDocs/data/"
CSV_DATA = "housing.data"

df = pd.read_csv(PATH + CSV_DATA, header=None)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values
# split into input (X) and output (Y) variables
X       = dataset[:, 0:13]
y       = dataset[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# define base model
from keras.optimizers import Adam   #for adam optimizer
def baseline_model():
    model = Sequential()
    model.add(Dense(25, input_dim=13, kernel_initializer='uniform',
                        activation='softplus'))
    model.add(Dense(10, kernel_initializer='lecun_uniform', activation='softplus'))
    model.add(Dense(1,  kernel_initializer='uniform'))

    # Use Adam optimizer with the given learning rate
    opt = Adam(lr=0.005)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100,
                           batch_size=9, verbose=1)
kfold   = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
print("Baseline RMSE: " + str(np.sqrt(results.std())))

# So then we build the model.
model = baseline_model()
history = model.fit(X_train, y_train, epochs=100,
                    batch_size=9, verbose=1,
                    validation_data=(X_test, y_test))
predictions = model.predict(X_test)

mse = metrics.mean_squared_error(y_test, predictions)
print("Neural network MSE: " + str(mse))
print("Neural network RMSE: " + str(np.sqrt(mse)))
