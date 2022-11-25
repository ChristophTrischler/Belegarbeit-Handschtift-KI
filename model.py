import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization, Dropout, GlobalAvgPool2D,\
    Activation, Input
from keras.applications import ResNet50, ResNet50V2
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt


nums = [x for x in range(10)]  # array of the nums how they should be
Labels = nums.copy()
abc = [chr(ord('A')+i) for i in range(26)]
Labels.extend(abc)

should = np.array(nums)
Labels = np.array(Labels)


def load_az_dataset(datasetPath):
    print("...")
    data = []
    labels = []

    for row in open(datasetPath):  # Openfile and start reading each row
        row = row.split(",")

        # row[0] contains label
        label = int(row[0])

        # Other all collumns contains pixel values make a saperate array for that
        image = np.array([x for x in row[1:]], dtype="uint8")

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

    return data, labels


def getData():
    # load data from tensorflow framework
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Stack train data and test data to form single array
    mnist_data = np.vstack([x_train, x_test])

    # Horizontal stacking labels of train and test set
    mnist_labels = np.hstack([y_train, y_test])

    az_data, az_labels = load_az_dataset("A_ZHandwrittenData.csv")

    # the MNIST dataset occupies the labels 0-9, so let's add 10 to every A-Z label to ensure the A-Z characters are
    # not incorrectly labeled

    az_labels += 10

    # stack the A-Z data and labels with the MNIST digits data and labels

    data = np.vstack([az_data, mnist_data])
    labels = np.hstack([az_labels, mnist_labels])

    data = [cv2.resize(image, (32, 32)) for image in data]  # reshape to 32px * 32px for ResNet50
    data = np.array(data, dtype="float32")
    data = np.expand_dims(data, axis=-1)
    data /= 255.0  # from int 0 to 255 to float 0 to 1

    """le = LabelBinarizer()
    labels = le.fit_transform(labels)"""

    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.20, stratify=labels,
                                                          random_state=42)

    return x_train, y_train, x_test, y_test


def createModel(data=None):
    if not data:
        data = getData()
    x_train, y_train, x_test, y_test = data

    model = Sequential()
    model.add(
        ResNet50V2(
            include_top=False,
            input_shape=(32, 32, 1),
            weights=None,
            pooling="avg"
        )
    )
    model.add(Dense(units=36, activation='softmax'))  # output layer

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


def testImgs(imgs: np.array, model: Model):
    predictions = model.predict(imgs)  # Model gives percentages to the numbers
    predictions = [np.argmax(prediction) for prediction in predictions]  # numbers with the highest percentages
    # corrects = (predictions == should) # array with 1's (true) by correct predictions and 0's by incorrect predictions
    # print(f"p: {predictions} s: {should} c: {corrects}")
    # accuracy = corrects.sum() / len(corrects) * 100  # accuracy in percentage
    # print(f"a: {accuracy}")
    predictions = [Labels[p] for p in predictions]
    return predictions  # , accuracy


def testImg(img: np.array, model: Model):
    prediction = model.predict(np.array([img]))  # Model gives percentages to the numbers
    return np.argmax(prediction[0])


def loadModel():
    model = keras.models.load_model("model")
    return model


def saveModel(model: keras.Model):
    model.save("model")


def main():
    data = getData()
    model = createModel(data)
    testModel(model, data)
    saveModel(model)


if __name__ == "__main__":
    main()
