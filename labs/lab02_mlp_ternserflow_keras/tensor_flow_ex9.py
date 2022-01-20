import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from global_constants import PATH

df = pd.read_csv(PATH + 'iris_old.csv')
df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']
print(df)

# Convert text to numeric category.
# 0 is setosa, 1 is versacolor and 2 is virginica
df['y'] = LabelEncoder().fit_transform(df['Iris Type'])

# Prepare the data.
dfX = df.iloc[:, 0:4]  # Get X features only from columns 0 to 3
dfY = df.iloc[:, 5:6]  # Get X features only from column 5

ROW_DIM = 0
COL_DIM = 1

x_array = dfX.values
x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
                                  x_array.shape[COL_DIM])

# Convert DataFrame columns to vertical columns of target variables values.
y_array = dfY.values
y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM],
                                  y_array.shape[COL_DIM])

X_train, X_test, y_train, y_test = train_test_split(
    x_arrayReshaped, y_arrayReshaped, test_size=0.33)

# Build a network model of sequential layers.
model = Sequential()

# Add 1st hidden layer. Note 1st hidden layer also receives data from input layer.
# The input array must contain two feature columns and any number of rows.
model.add(Dense(12, activation='sigmoid',
                input_shape=(x_arrayReshaped.shape[COL_DIM],)))

# Add output layer.
model.add(Dense(3, activation='softmax'))

# Compile the model.
# Binary cross entropy is used to measure error cost for binary predictions.
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
# An epoch is one iteration for all samples through the network.
# verbose can be set to 1 to show detailed output during training.
model.fit(x_arrayReshaped, y_arrayReshaped, epochs=1000, verbose=1)

# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# Make a prediction
row = [5.1, 3.5, 1.4, 0.2]
yhat = model.predict([row])
print(yhat)
