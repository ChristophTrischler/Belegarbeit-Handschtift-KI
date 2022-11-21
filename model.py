import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization, Dropout, GlobalAvgPool2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt


should = np.array([x for x in range(10)])  # array of the nums how they should be


def getData():
    ydata = []
    xdata = []

    for row in open("A_ZHandwrittenData.csv", "r"):
        # in csv first colum label rest image pxs
        row = row.split(",")
        ydata.append(int(row[0]) + 10)  # offset of 10 <= 0-9 are the numbers => 10-35 Letters

        img = np.array([int(i) for i in row[1:]]).reshape((28, 28))
        xdata.append(img)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # load numberdata from keras

    xdata.extend(x_test)
    xdata.extend(x_train)
    ydata.extend(y_train)
    ydata.extend(y_test)

    ydata = np.array(ydata,  dtype=np.uint8)
    xdata = np.array(xdata, dtype=np.uint8)

    xdata = [cv2.threshold(x, 50, 255, cv2.THRESH_BINARY)[1] for x in xdata]
    xdata = keras.utils.normalize(xdata)

    x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, random_state=420)

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
