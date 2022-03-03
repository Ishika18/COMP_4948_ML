import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from global_constants import PATH

PATH       = PATH + "small_cats_and_dogs_and_octopi/"

# Loads the image file.
def load_image(fileName, subFolder):
    try:
      folder = PATH + subFolder + "/"
      filePath = folder+ fileName
      img = cv2.imread(filePath)
      return img
    except:
      return np.NaN

# Extract the actual class from the image file name.
def extract_label(file_name):
    if "dog" in file_name:
        return 1
    elif "cat" in file_name:
        return 0
    else:
        return 2


# Transforms image into scaled and squared image.
def preprocess_image(img):
    IMAGE_SIZE = 96

    try:
      # Finds minimum of height and width.
      min_side = min(img.shape[0], img.shape[1])

      # Reduce image to square using minimum of width and height.
      img = img[:min_side, :min_side]

      # Reside to 96x96.
      img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
      print(img.shape)

      # Eliminate three byte color channel.
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      print(img.shape)
      print("***")

      # Scale numbers to range between 0 and 1.0
      return img / 255.0
    except:
      return np.NaN

def getImages(dirName):
    imageList   = []
    labelList   = []

    # List all items in the directory.
    image_files = os.listdir(PATH + dirName + "/")

    for i in range(0, len(image_files)):
        image = load_image(image_files[i], dirName)
        processedImage = preprocess_image(image)

        # Build list of processed images and labels.
        if processedImage is not np.NaN:
            imageList.append(processedImage)

            label = extract_label(image_files[i])
            labelList.append(label)
    return imageList, labelList

# Displays grayscale image and image data to ensure everything is working properly.
def verifyImage(images, labels, index):
    print("Image Label: " + str(labels[index]))
    print("Image Shape: " + str(images[index].shape))
    print("Image Data: ")
    print(images[index])

    # Display the image.
    plt.imshow(images[index])
    plt.show()
    print("*****")

# Add an extra dimension for the signle color channel.
# Changes (2000, 96, 96) to (2000, 96, 96, 1)

# Load transformed  and scaled images.
X_train, y_train = getImages('train')
X_test, y_test   = getImages('test')

# Verify first image in training set.
verifyImage(X_train, y_train, 0)

# Verify second image in training set.
verifyImage(X_train, y_train, 1)

def convertToArray(x, y):
    x = np.array(x)
    x = np.expand_dims(x, -1)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    print("***")
    return x, y

X_train, y_train = convertToArray(X_train, y_train)
X_test, y_test   = convertToArray(X_test, y_test)


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

def create_model(X_train, y_train, X_test, y_test):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=X_train.shape[1:]))
    # Add a pooling layer to down sample height and widths by half.
    # pool_size is filter size.
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # It is common to repeat this pattern a few times by doubling the filter
    # each time to offset the reduction in size from down sampling.
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))


    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # flatten????
    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))

    # Create output layer which has two classes.
    model.add(Dense(units=3, activation='softmax'))

    model.compile(optimizer=tf.optimizers.Adam(),
        # We are using classification.
        loss=tf.losses.SparseCategoricalCrossentropy(),
        # Show accuracy.
        metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=100,
                        validation_data=(X_test, y_test))
    model.save_weights("model.tf")
    return model, history

model, history = create_model(X_train, y_train, X_test, y_test)

predictions = model.predict(X_test)

# Iterates through pairs of predictions and adds most probable option to list.
predictionList = []

# Iterates through pairs of predictions and adds most probable option to list.
for i in range(0, len(predictions)):
    prediction =  predictions[i]
    if(prediction[0] > prediction[1]):
        predictionList.append(0)
    else:
        predictionList.append(1)

predictionArray = np.array(predictionList)

print(predictionList)
import pandas as pd
from sklearn import metrics

# Show confusion matrix and accuracy scores.
confusion_matrix = pd.crosstab(y_test, predictionArray,
                               rownames=['Actual'],
                               colnames=['Predicted'])

print('\nAccuracy: ',metrics.accuracy_score(y_test, predictionArray))
print("\nConfusion Matrix")
print(confusion_matrix)

