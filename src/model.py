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
import cv2
import numpy as np
from matplotlib import pyplot as plt


nums = [x for x in range(10)]  # array of the nums how they should be
Labels = nums.copy()    # create Label translation
abc = [chr(ord('A')+i) for i in range(26)]
Labels.extend(abc)


should = np.array(nums)
Labels = np.array(Labels)


def loadCsvDataset(datasetPath="src/A_ZHandwrittenData.csv"):
    print("...")
    data = []
    labels = []

    for row in open(datasetPath):
        # in csv row 0 for labels rest image data
        row = row.split(",")

        label = int(row[0])

        image = np.array(row[1:], dtype="uint8")
        image = image.reshape((28, 28))  # reshape from 1d array to 2d array

        # can be used show letters from the Dataset
        """if label == ord('k'):
            plt.imshow(image)
            plt.show()"""

        data.append(image)
        labels.append(label)

    data = np.array(data, dtype='float32')
    labels = np.array(labels)

    return data, labels


def getData():
    # load 0-9 data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Stack train data and test data to form single array
    mnist_data = np.vstack([x_train, x_test])

    # Horizontal stacking labels of train and test set
    mnist_labels = np.hstack([y_train, y_test])

    az_data, az_labels = loadCsvDataset()

    # 0-9 -> numbers 10-36 -> ABC
    az_labels += 10  # ABC: 0-26 -> 10-36

    # stack ABC and numbers

    data = np.vstack([az_data, mnist_data])
    labels = np.hstack([az_labels, mnist_labels])

    data = [cv2.resize(image, (32, 32)) for image in data]  # reshape to 32px * 32px for ResNet50
    data = np.array(data, dtype="float32")
    data = np.expand_dims(data, axis=-1)
    data /= 255.0  # from int 0 to 255 to float 0 to 1

    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.20, stratify=labels,
                                                          random_state=42)

    return x_train, y_train, x_test, y_test


def createModel(data=None):
    if not data:
        data = getData()
    x_train, y_train, x_test, y_test = data

    model = Sequential()
    model.add(
        ResNet50(
            include_top=False,
            input_shape=(32, 32, 1),
            weights=None,
            pooling="avg"
        )
    )
    model.add(Dense(units=72, activation="relu"))
    model.add(Dense(units=36, activation="softmax"))  # output layer

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20)
    return model


def testModel(model, data=None):
    if not data:
        data = getData()
    x_train, y_train, x_test, y_test = data
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(f"acc: {val_acc}")
    print(f"loss: {val_loss}")


def testImgs(imgs: np.array, model: Model):
    predictions = model.predict(imgs, use_multiprocessing=True )  # Model gives percentages to the numbers
    predictions = [np.argmax(prediction) for prediction in predictions]  # numbers with the highest percentages
    predictions = [Labels[p] for p in predictions]
    return predictions


def testImg(img: np.array, model: Model):
    prediction = model.predict(np.array([img]))  # Model gives percentages to the numbers
    return np.argmax(prediction[0])


def loadModel():
    model = keras.models.load_model("model")
    return model


def main():
    data = getData()
    model = createModel(data)
    testModel(model, data)
    model.save("model")


if __name__ == "__main__":
    main()
