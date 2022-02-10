from keras.models     import Sequential
from keras.layers     import Dense
from os               import makedirs
from sklearn.preprocessing   import LabelEncoder
from os import path
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf

import global_constants

PATH = './models/'

# fit model on dataset
def fit_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model


def generateData():
    df = pd.read_csv(global_constants.PATH + 'iris_old.csv')
    df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']

    # Convert text to numeric category.
    # 0 is setosa, 1 is versacolor and 2 is virginica
    df['y'] = LabelEncoder().fit_transform(df['Iris Type'])

    # Prepare the data.
    X = df[['Sepal L', 'Sepal W', 'Petal L', 'Petal W']]
    y = df['y']
    ROW_DIM = 0
    COL_DIM = 1

    x_array = X.values
    x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
                                      x_array.shape[COL_DIM])

    y_array = y.values
    y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM], 1)

    trainX, testX, trainy, testy = train_test_split(x_arrayReshaped,
                                                    y_arrayReshaped,
                                                    test_size=0.80)
    return trainX, testX, trainy, testy


def generateModels(trainX, trainy):
    # create directory for models
    if(not path.exists(PATH)):
        makedirs('./models')

    # fit and save models
    numModels = 5
    for i in range(numModels):
        # fit model
        model = fit_model(trainX, trainy)
        # save model
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        model.save(filename)
        print('>Saved %s' % filename)

trainX, testX, trainy, testy = generateData()

# one hot encode output variable
trainy = tf.keras.utils.to_categorical(trainy)
generateModels(trainX, trainy)

# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of models
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

trainX, testX, trainy, testy = generateData()

# load all models
numModels = 5
models    = load_all_models(numModels)
print('Loaded %d models' % len(models))

# evaluate standalone models on test dataset
# individual ANN models are built with one-hot encoded data.
for model in models:
    oneHotEncodedY = tf.keras.utils.to_categorical(testy)
    _, acc = model.evaluate(testX, oneHotEncodedY, verbose=0)
    print('Model Accuracy: %.3f' % acc)


# create stacked model input dataset as outputs from ensemble
def getStackedData(models, inputX):
    stackXdf = None
    for model in models:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        singleModelPredDf = pd.DataFrame(np.row_stack(yhat))

        # Store predictions of all models for 1 sample in each df row.
        # Here is 1st row for 5 models with predictions for 3 classes each.
        # 5 models x 3 classes = 15 columns.
        #          0             1         2   ...        12            13        14
        # 0 0.993102  1.106366e-04  0.006788   ...  0.993102  1.106366e-04  0.006788
        if stackXdf is None:
            stackXdf = singleModelPredDf
        else:
            numClasses = len(singleModelPredDf.keys())
            numStackXCols = len(stackXdf.keys())

            # Add new classification columns.
            for i in range(0, numClasses):
                stackXdf[numStackXCols + i] = stackXdf[i]
    return stackXdf


# Make predictions with the stacked model
def stacked_prediction(models, model, inputX):
    # create dataset using ensemble
    stackedX = getStackedData(models, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

# fit a model based on the outputs from the ensemble models
def fit_stacked_model(models, inputX, inputy):
    # create dataset using ensemble
    stackedX = getStackedData(models, inputX)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model

# fit stacked model using the ensemble
# Stacked model build with LogisticRegression.
# y for LogisticRegression is not one-hot encoded.
model = fit_stacked_model(models, testX, testy)

# evaluate model on test set
yhat = stacked_prediction(models, model, testX)
acc  = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
