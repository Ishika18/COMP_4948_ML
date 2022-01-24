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

# Create vertical array of features.
x_array = dfX.values
x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
                                  x_array.shape[COL_DIM])

y_array = dfY.values
y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM],
                                  y_array.shape[COL_DIM])

# Split into train and test data sets.
X_train, X_test, y_train, y_test = train_test_split(
    x_arrayReshaped, y_arrayReshaped, test_size=0.33)

n_features = X_train.shape[COL_DIM]

# Define the model.
model = Sequential()

# Hidden layer 1 (also receives the input layer)
model.add(Dense(2, activation='relu', input_shape=(n_features,)))

# Output layer
model.add(Dense(3, activation='softmax'))

# Compile the model.
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model.
history = model.fit(X_train, y_train, epochs=1000, batch_size=28, verbose=1,
                    validation_data=(X_test, y_test))

# Evaluate the model.
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# make a prediction
row = [5.1, 3.5, 1.4, 0.2]
yhat = model.predict([row])
print('Predicted: s (class=d)' + str(yhat))

import matplotlib.pyplot as plt


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


def showAccuracy(history):
    # Get training and test loss histories
    training_loss = history.history['accuracy']
    validation_loss = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)
    # Visualize loss history for training data.
    plt.plot(epoch_count, training_loss, label='Train Accuracy', color='red')

    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, 'r--',
             label='Validation Accuracy', color='black')
    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.title('Accuracy')


plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
showLoss(history)
showAccuracy(history)
plt.show()
