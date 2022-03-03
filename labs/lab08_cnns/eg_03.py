
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
    model.add(Dense(units=2, activation='softmax'))

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
