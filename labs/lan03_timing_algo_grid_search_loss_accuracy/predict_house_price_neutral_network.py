import time

import pandas as pd
import numpy  as np
from sklearn                 import metrics
from sklearn.model_selection import train_test_split
from global_constants import PATH
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

ROW_DIM = 0
COL_DIM = 1
x_arrayReshaped = X.reshape(X.shape[ROW_DIM],
                            X.shape[COL_DIM])

# Convert DataFrame columns to vertical columns of target variables values.
y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
                     y_arrayReshaped, test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# define base model  #for adam optimizer
def baseline_model():
    model = Sequential()
    model.add(Dense(25, input_dim=13, kernel_initializer='uniform',
                        activation='softplus'))
    model.add(Dense(10, kernel_initializer='lecun_uniform', activation='softplus'))
    model.add(Dense(1,  kernel_initializer='uniform'))

    # Use Adam optimizer with the given learning rate
    opt = Adam(lr=0.005)
    model.compile(loss='mean_squared_error')
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
def currentTimeInMilliseconds():
    newTime = int(round(time.time() * 1000))
    return newTime


startTime = currentTimeInMilliseconds()
history = model.fit(X_train, y_train, epochs=100,
                    batch_size=9, verbose=1,
                    validation_data=(X_test, y_test))
endTime = currentTimeInMilliseconds()
predictions = model.predict(X_test)
print("Time taken to fit teh model: " + str(endTime - startTime) + " ms")

mse = metrics.mean_squared_error(y_test, predictions)
print("Neural network MSE: " + str(mse))
print("Neural network RMSE: " + str(np.sqrt(mse)))


def showLoss(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 1)
    # Visualize loss history for training data.
    plt.plot(epoch_count, training_loss, label='Train Loss', color='red')

    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
             color='black')

    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.title("Loss")
    plt.show()

showLoss(history)