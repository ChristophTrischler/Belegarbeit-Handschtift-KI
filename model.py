import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization, Dropout, GlobalAvgPool2D,\
    Activation, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt


should = np.array([x for x in range(10)])  # array of the nums how they should be


def load_az_dataset(datasetPath):
    # List for storing data
    data = []

    # List for storing labels
    labels = []

    for row in open(datasetPath):  # Openfile and start reading each row
        # Split the row at every comma
        row = row.split(",")

        # row[0] contains label
        label = int(row[0])

        # Other all collumns contains pixel values make a saperate array for that
        image = np.array([int(x) for x in row[1:]], dtype="uint8")

        # Reshaping image to 28 x 28 pixels
        image = image.reshape((28, 28))

        # append image to data
        data.append(image)

        # append label to labels
        labels.append(label)

    # Converting data to numpy array of type float32
    data = np.array(data, dtype='float32')

    # Converting labels to type int
    labels = np.array(labels, dtype="int")

    return (data, labels)


def getData():
    # load data from tensorflow framework
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Stack train data and test data to form single array
    mnist_data = np.vstack([x_train, x_test])

    # Horizontal stacking labels of train and test set
    mnist_labels = np.hstack([y_train, y_test])

    # Uniques and counts of train labels
    #unique_train, counts_train = np.unique(y_train, return_counts=True)
    #print(f"Value counts of y_train modalities: {counts_train}\n")

    # Uniques and counts of test labels
    #unique_test, counts_test = np.unique(y_test, return_counts=True)
    #print(f"Value counts of y_test modalities: {counts_test}")

    az_data, az_labels = load_az_dataset("A_ZHandwrittenData.csv")

    # the MNIST dataset occupies the labels 0-9, so let's add 10 to every A-Z label to ensure the A-Z characters are not incorrectly labeled

    az_labels += 10

    # stack the A-Z data and labels with the MNIST digits data and labels

    data = np.vstack([az_data, mnist_data])
    labels = np.hstack([az_labels, mnist_labels])

    # add a channel dimension to every image in the dataset and scale the
    # pixel intensities of the images from [0, 255] down to [0, 1]

    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    """le = LabelBinarizer()
    labels = le.fit_transform(labels)

    counts = labels.sum(axis=0)

    # account for skew in the labeled data
    classTotals = labels.sum(axis=0)
    classWeight = {}

    # loop over all classes and calculate the class weight
    for i in range(0, len(classTotals)):
        classWeight[i] = classTotals.max() / classTotals[i]"""

    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

    return x_train, y_train, x_test, y_test


def createModel(data=None):
    if not data:
        data = getData()
    x_train, y_train, x_test, y_test = data

    """model = Sequential()
    model.add(Flatten())  # Input Layer
    model.add(Dense(512, activation='relu'))  # Hidden Layer 1
    model.add(Dense(256, activation='relu'))  # Hidden Layer 1
    model.add(Dense(128, activation='relu'))  # Hidden Layer 2
    model.add(Dense(128, activation='relu'))  # Hidden Layer 3"""
    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=36, activation='softmax'))

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50)
    return model


def testModel(model, data=None):
    if not data:
        data = getData()
    x_train, y_train, x_test, y_test = data
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(f"acc: {val_acc}")
    print(f"loss: {val_loss}")


def testNumImg(imgs: np.array, model: Model):
    """reshapedImg = imgs.reshape(-1, 784)  # 3d Array ( length * 28 * 28 ) to 2d ( length * 784 )"""
    predictions = model.predict(imgs)  # Model gives percentages to the numbers
    predictions = [np.argmax(prediction) for prediction in predictions]  # numbers with the highest percentages
    corrects = (predictions == should)  # array with 1's (true) by correct predictions and 0's by incorrect predictions
    print(f"p: {predictions} s: {should} c: {corrects}")
    accuracy = corrects.sum() / len(corrects) * 100  # accuracy in percentage
    print(f"a: {accuracy}")
    return predictions, accuracy


def testImg(img: np.array, model: Model):
    prediction = model.predict(np.array([img]))  # Model gives percentages to the numbers
    return np.argmax(prediction[0])


def loadModel():
    model = keras.models.load_model("model")
    return model


def saveModel(model: keras.Model):
    model.save("model")


def main():
    data = getData()  # getData()

    model = createModel(data)
    testModel(model, data)
    saveModel(model)


if __name__ == "__main__":
    main()
