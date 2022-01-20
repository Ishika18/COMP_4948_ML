import pandas as pd
import numpy  as np
from sklearn                 import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.models            import Sequential
from keras.layers            import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from global_constants import PATH
from tensorflow.keras.optimizers import Adam

CSV_DATA     = "heart_disease.csv"
FILE     = "heart_disease.csv"
data     = pd.read_csv(PATH + FILE)
x_array   = data.drop("target", axis=1)
y_array = data["target"]


ROW_DIM = 0
COL_DIM = 1

X_train, X_test, y_train, y_test = train_test_split(
    x_array, y_array, test_size=0.3, random_state=42
)

# Build a network model of sequential layers.
model = Sequential()


# Add 1st hidden layer. Note 1st hidden layer also receives data from input layer.
# The input array must contain two feature columns and any number of rows.

# Add 1st hidden layer. Note 1st hidden layer also receives data from input layer.
# The input array must contain two feature columns and any number of rows.
model.add(Dense(20, kernel_initializer='uniform',
                    input_dim=x_array.shape[COL_DIM], activation='tanh'))

# Add output layer.
model.add(Dense(1, activation='sigmoid'))

# Compile the model.
# Binary cross entropy is used to measure error cost for binary predictions.
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
# An epoch is one iteration for all samples through the network.
# verbose can be set to 1 to show detailed output during training.
model.fit(x_array, y_array, epochs=1000, verbose=1)

# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
