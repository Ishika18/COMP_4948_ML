import pandas as pd
import numpy  as np
from sklearn                 import metrics
from sklearn.model_selection import train_test_split
from keras.models            import Sequential
from keras.layers            import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from global_constants import PATH
from tensorflow.keras.optimizers import Adam

CSV_DATA = "housing.data"
df       = pd.read_csv(PATH + CSV_DATA,  header=None)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())
print(df.tail())
print(df.describe())

# Convert DataFrame columns to vertical columns so they can be used by the NN.
dataset = df.values
X       = dataset[:, 0:13]  # Columns 0 to 12
y       = dataset[:, 13]    # Columns 13
ROW_DIM = 0
COL_DIM = 1

x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
y_arrayReshaped = y.reshape(y.shape[ROW_DIM],1)

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
         y_arrayReshaped, test_size=0.2, random_state=0)


# Define the model.
def create_model(learningRate=0.001):
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Use Adam optimizer with the given learning rate
    opt = Adam(lr=learningRate)
    model.compile(loss='mse', metrics=['accuracy'], optimizer=opt)
    return model

# Since this is a linear regression use KerasRegressor.
estimator = KerasRegressor(build_fn=create_model, epochs=100,
                           batch_size=10, verbose=1)

# Use kfold analysis for a more reliable estimate.
kfold   = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
print("Baseline RMSE: " + str(np.sqrt(results.std())))

# Build the model.
model   = create_model()
history = model.fit(X_train, y_train, epochs=100,
                    batch_size=10, verbose=1,
                    validation_data=(X_test, y_test))

# Evaluate the model.
predictions = model.predict(X_test)
mse         = metrics.mean_squared_error(y_test, predictions)
print("Neural network MSE: " + str(mse))
print("Neural network RMSE: " + str(np.sqrt(mse)))
