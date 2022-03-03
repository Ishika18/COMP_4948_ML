import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from global_constants import PATH

PATH       = PATH + "small_cats_and_dogs/"

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
    return 1 if "dog" in file_name else 0

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


# Verify first image in training set.
verifyImage(X_train, y_train, 0)

# Verify second image in training set.
verifyImage(X_train, y_train, 1)
