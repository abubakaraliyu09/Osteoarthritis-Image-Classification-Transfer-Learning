
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Softmax, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Activation, add
from keras.layers import concatenate
from keras.losses import CategoricalCrossentropy
from keras.optimizers import RMSprop, Adam, Adamax, Nadam, SGD
from keras import Model
#from keras.optimizers.schedules import InverseTimeDecay, ExponentialDecay
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import os
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from copy import deepcopy

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))



#Preprocessing

path = './aug'


def encode_labels(labels):
    labels = np.where(labels == 'Doubtful', 0, labels)
    labels = np.where(labels == 'Healthy', 1, labels)
    labels = np.where(labels == 'Minimal', 2, labels)
    labels = np.where(labels == 'Moderate', 3, labels)
    labels = np.where(labels == 'Severe', 4, labels)
    return labels


def fix_images(images, avgHeight, avgWidth):
    temp = images.copy()
    for i in range(len(temp)):
        temp[i] = cv.resize(temp[i], (int(avgWidth), int(avgHeight)))
    return temp


def convertTo4Class(arr):
    tmp = np.zeros((arr.shape[0], 5))
    tmp[np.where(arr == '0')[0], :] = [1, 0, 0, 0, 0]
    tmp[np.where(arr == '1')[0], :] = [0, 1, 0, 0, 0]
    tmp[np.where(arr == '2')[0], :] = [0, 0, 1, 0, 0]
    tmp[np.where(arr == '3')[0], :] = [0, 0, 0, 1, 0]
    tmp[np.where(arr == '4')[0], :] = [0, 0, 0, 0, 1]
    return tmp


# function for convert probablity to 0s and 1s for multiple classification
def argmaxKeepDimensions(arr):
    tmp = np.zeros_like(arr)
    tmp[np.arange(len(arr)), arr.argmax(1)] = 1
    return tmp


images, labels = [], []
avgHeight, avgWidth = 0, 0
for filename in os.listdir(path):
    for filename1 in os.listdir(path + '/' + filename):
        img = cv.imread(path + '/' + filename + '/' + filename1)
        if img is not None:
            images.append(img)
            # print(img.shape)
            labels.append(filename)

image_width = 100
image_height = 75

# ------------------------------------- fix input images -------------------------------------
fixed_images = fix_images(images, image_height, image_width)
fixed_images, labels = np.array(fixed_images), np.array(labels)
fixed_images = fixed_images / 255

# ------------------------------------- fix output labels -------------------------------------
fixed_labels = labels.reshape(labels.shape[0], 1)
fixed_labels = encode_labels(fixed_labels)
fixed_labels = convertTo4Class(fixed_labels)
print(fixed_images.shape)
print(fixed_labels.shape)



X_train, X_test, y_train, y_test = train_test_split(fixed_images, fixed_labels, test_size=0.2, random_state=20)
batch_size = X_train.shape[0] // 2
# horizontal flip
horizontal_flip = ImageDataGenerator(horizontal_flip=True, rescale=1. / 255)
# 40 degrees rotations
rotation = ImageDataGenerator(rotation_range=40, rescale=1. / 255)
x, y = next(horizontal_flip.flow_from_directory(path, target_size=(75, 100), batch_size=600))
X_train = np.concatenate((X_train, x), axis=0)
y_train = np.concatenate((y_train, y), axis=0)


x, y = next(rotation.flow_from_directory(path, target_size=(75, 100), batch_size=400))
X_train = np.concatenate((X_train, x), axis=0)
y_train = np.concatenate((y_train, y), axis=0)

print("X_train.shape = ", X_train.shape)
print("y_train.shape = ", y_train.shape)
print("X_test.shape = ", X_test.shape)
print("y_test.shape = ", y_test.shape)


'''
Implementation-
Neural Network
'''

class NN:

    def train(self, X_train, y_train, validation_data, optimizer, epoch):
        self.X_train = X_train
        self.y_train = y_train

        self.model.compile(loss=CategoricalCrossentropy(),
                           optimizer=optimizer, metrics=['accuracy'])

        # best model based on validation accuracy will save in this path
        filepath = 'my_best_model.hdf5'
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='max')

        self.history = self.model.fit(X_train, y_train, validation_data=validation_data,
                                      batch_size=3, epochs=epoch, verbose=1, callbacks=[checkpoint]).history

        # best model assign to final model
        self.model = load_model(filepath)

    def evaluate(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.test_result = self.model.evaluate(X_test, y_test, verbose=0)

    def outputResult(self):
        # print("result in training set:", "\ntrain acc= ",
        #       self.history['accuracy'], "\ntrain loss= ", self.history['loss'])
        # print("\nresult in test set:", "\ntest acc= ",
        #       self.history['val_accuracy'], "\ntest loss= ", self.history['val_loss'], '\n')

        # best models based on acc or loss in tarin set or test set
        trainHistory = list(
            map(lambda x, y: [x, y], self.history['accuracy'], self.history['loss']))
        testHistory = list(
            map(lambda x, y: [x, y], self.history['val_accuracy'], self.history['val_loss']))
        print(
            f"\nbest model based on min training set loss:  acc= {min(trainHistory, key=lambda k: k[1])[0]}  loss= {min(trainHistory, key=lambda k: k[1])[1]}")
        print(
            f"best model based on min test set loss:  acc= {min(testHistory, key=lambda k: k[1])[0]}  loss= {min(testHistory, key=lambda k: k[1])[1]}")
        print(
            f"best model based on max training set accuracy:  acc= {max(trainHistory, key=lambda k: k[0])[0]}  loss= {max(trainHistory, key=lambda k: k[0])[1]}")
        print(
            f"best model based on max test set accuracy:  acc= {max(testHistory, key=lambda k: k[0])[0]}  loss= {max(testHistory, key=lambda k: k[0])[1]}")

        print("\nevaluate dataset with best model based on maximum test set accuracy")
        print("evaluate train set= ", self.model.evaluate(
            self.X_train, self.y_train, verbose=0))
        print("evaluate test set= ", self.test_result)

        y_train_pred = self.model.predict(self.X_train)
        # convert probablities to 0s and 1s
        y_train_pred = argmaxKeepDimensions(y_train_pred)

        y_test_pred = self.model.predict(self.X_test)
        # convert probablities to 0s and 1s
        y_test_pred = argmaxKeepDimensions(y_test_pred)

        # confusion matrix and precision, recall and f1 report
        print('\n', '-' * 30, 'metrics for traning set', '-' * 30)
        print("confusion matrix: \n", metrics.confusion_matrix(
            np.argmax(self.y_train, axis=1), np.argmax(y_train_pred, axis=1)))
        print(metrics.classification_report(self.y_train,
                                            y_train_pred, digits=3,
                                            target_names=['Cloudy', 'Rain', 'Shine', 'Sunrise']))

        # confusion matrix and precision, recall and f1 report
        print('-' * 30, 'metrics for test set', '-' * 30)
        print("confusion matrix: \n", metrics.confusion_matrix(
            np.argmax(self.y_test, axis=1), np.argmax(y_test_pred, axis=1)))
        print(metrics.classification_report(self.y_test,
                                            y_test_pred, digits=3, target_names=['Cloudy', 'Rain', 'Shine', 'Sunrise']))

        # self.model.summary()

    def showPlots(self):
        plt.plot(self.history['accuracy'],
                 label='training accuracy', marker='.', color='green')
        plt.plot(self.history['val_accuracy'],
                 label='test accuracy', marker='.', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(self.history['loss'],
                 label='training loss', marker='.', color='green')
        plt.plot(self.history['val_loss'],
                 label='test loss', marker='.', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



'''
Dense Neural Network
'''
class DNN(NN):

    def __init__(self, activiation, layer, dropout=False, batch_norm=False):
        self.model = Sequential()

        # Input layer
        self.model.add(Input(shape=(image_height, image_width, 3)))
        self.model.add(Flatten())
        if dropout:
            self.model.add(Dropout(0.2))
        elif batch_norm:
            self.model.add(BatchNormalization())

        # Hidden layers
        for i in range(layer - 1):
            self.model.add(
                Dense(2 ** (layer - i + 2), activation=activiation))
            if dropout:
                self.model.add(Dropout(0.2))
            elif batch_norm:
                self.model.add(BatchNormalization())

        # Output layer
        self.model.add(Dense(5, activation='softmax'))

'''
Convolutional Neural Network
'''
class CNN(NN):

    def __init__(self, activation, model_number=1):
        self.model = Sequential()

        # --------- Input layer ---------
        self.model.add(Input(shape=(image_height, image_width, 3)))

        # --------- Models ---------
        if model_number == 1:
            self.model1(activation)
        elif model_number == 2:
            self.model2(activation)
        elif model_number == 3:
            self.model3(activation)
        elif model_number == 4:
            self.dropout_model1(activation)
        elif model_number == 5:
            self.dropout_model2(activation)
        elif model_number == 6:
            self.dropout_model3(activation)
        elif model_number == 7:
            self.pooling_model1(activation)
        elif model_number == 8:
            self.pooling_model2(activation)
        elif model_number == 9:
            self.pooling_model3(activation)
        elif model_number == 10:
            self.pooling_model4(activation)
        elif model_number == 11:
            self.pooling_model5(activation)
        elif model_number == 12:
            self.pooling_model6(activation)

        # --------- Output layer ---------
        self.model.add(Dense(4, activation='softmax'))

    def model1(self, activation):
        # Convolutional Layers
        self.model.add(Conv2D(16, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, 4, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(64, activation=activation))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation=activation))
        self.model.add(BatchNormalization())
        self.model.add(Dense(16, activation=activation))
        self.model.add(BatchNormalization())

    def model2(self, activation):
        # Convolutional Layers
        self.model.add(Conv2D(16, 4, 1, padding="valid", dilation_rate=2, activation=activation))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, 4, 1, padding="valid", dilation_rate=2, activation=activation))
        self.model.add(BatchNormalization())

        # self.model.add(Conv2D(64, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        # self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        # self.model.add(Dense(64, activation=activation))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation=activation))
        self.model.add(BatchNormalization())

    def model3(self, activation):

        # Convolutional Layers
        self.model.add(Conv2D(32, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(128, activation=activation))
        self.model.add(BatchNormalization())

    def dropout_model1(self, activation):

        # Convolutional Layers
        self.model.add(Conv2D(64, 3, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, 5, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(Dropout(0.2))

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(128, activation=activation))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(16, activation=activation))
        self.model.add(Dropout(0.2))

    def dropout_model2(self, activation):
        # Convolutional Layers
        self.model.add(Conv2D(16, 4, 1, padding="valid", dilation_rate=2, activation=activation))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, 4, 1, padding="valid", dilation_rate=2, activation=activation))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation=activation))
        self.model.add(Dropout(0.2))

    def dropout_model3(self, activation):
        # Convolutional Layers
        self.model.add(Conv2D(32, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation=activation))
        self.model.add(Dropout(0.2))

    def pooling_model1(self, activation):
        # Convolutional Layers
        self.model.add(Conv2D(16, 3, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=2, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, 3, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=2, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(64, activation=activation))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation=activation))
        self.model.add(BatchNormalization())

    def pooling_model2(self, activation):
        # Convolutional Layers
        self.model.add(Conv2D(16, 4, 1, padding="valid", dilation_rate=2, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, 4, 1, padding="valid", dilation_rate=2, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(32, activation=activation))
        self.model.add(BatchNormalization())

    def pooling_model3(self, activation):

        # Convolutional Layers
        self.model.add(Conv2D(32, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(128, activation=activation))
        self.model.add(BatchNormalization())

    def pooling_model4(self, activation):

        # Convolutional Layers
        self.model.add(Conv2D(32, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(MaxPooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(MaxPooling2D((2, 2), strides=2, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(128, activation=activation))
        self.model.add(BatchNormalization())
        # self.model.add(Dense(32, activation=activation))
        # self.model.add(BatchNormalization())

    def pooling_model5(self, activation):

        # Convolutional Layers
        self.model.add(Conv2D(64, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(MaxPooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(MaxPooling2D((2, 2), strides=2, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(256, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dense(256, activation=activation))
        self.model.add(BatchNormalization())
        # self.model.add(Dense(64, activation=activation))
        # self.model.add(BatchNormalization())

    def pooling_model6(self, activation):

        # Convolutional Layers
        self.model.add(Conv2D(64, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(MaxPooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, 4, 1, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(MaxPooling2D((2, 2), strides=2, padding="valid"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(256, 2, 2, padding="valid", dilation_rate=1, activation=activation))
        self.model.add(AveragePooling2D((2, 2), strides=1, padding="valid"))
        self.model.add(BatchNormalization())

        # Dense Layers
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256, activation=activation))
        self.model.add(Dropout(0.2))

'''
Inception Network
'''
class InceptionNet(NN):

    def __init__(self, activation, model_number=1):
        # Input layer
        input = Input(shape=(image_height, image_width, 3))

        # --------- Models ---------
        if model_number == 1:
            x = self.model1(input, activation)
        elif model_number == 2:
            x = self.model2(input, activation)
        elif model_number == 3:
            x = self.model3(input, activation)
        elif model_number == 4:
            x = self.model4(input, activation)

        # Output layer
        output = Dense(4, activation='softmax')(x)

        self.model = Model(input, output, name='Inception_v3')

    # function for creating a projected inception module
    def inception_module(self, layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        # 1x1 conv
        # x = self.inception_module(input, 8, 16, 32, 4, 8, 8)

        conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
        # conv1 = BatchNormalization()(conv1)
        # 3x3 conv
        conv3 = Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
        conv3 = Conv2D(f2_out, (3, 3), padding='same', activation='relu')(conv3)
        # conv3 = BatchNormalization()(conv3)
        # 5x5 conv
        conv5 = Conv2D(f3_in, (1, 1), padding='same', activation='relu')(layer_in)
        conv5 = Conv2D(f3_out, (5, 5), padding='same', activation='relu')(conv5)
        # conv5 = BatchNormalization()(conv5)

        # 3x3 max pooling
        pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
        pool = Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return layer_out

    def naive_inception_module(self, layer_in, f1, f2, f3):
        # 1x1 conv
        conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
        # 3x3 conv
        conv3 = Conv2D(f2, (3, 3), padding='same', activation='relu')(layer_in)
        # 5x5 conv
        conv5 = Conv2D(f3, (5, 5), padding='same', activation='relu')(layer_in)
        # 3x3 max pooling
        pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return layer_out

    def model1(self, input, activation):
        # add inception block 1
        # x = self.inception_module(input, 64, 96, 128, 16, 32, 32)
        x = self.inception_module(input, 8, 16, 32, 4, 8, 8)
        x = BatchNormalization()(x)

        # add inception block 2
        x = self.inception_module(x, 32, 32, 64, 8, 16, 8)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(64, activation=activation)(x)

        return x

    def model2(self, input, activation):
        # add inception block 1
        # x = self.inception_module(input, 64, 96, 128, 16, 32, 32)
        x = self.inception_module(input, 8, 16, 32, 4, 8, 8)
        x = BatchNormalization()(x)

        # add inception block 2
        x = self.inception_module(x, 8, 16, 32, 4, 8, 8)
        x = BatchNormalization()(x)

        # add inception block 3
        x = self.inception_module(x, 32, 32, 64, 8, 16, 8)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(64, activation=activation)(x)

        return x

    def model3(self, input, activation):
        # add inception block 1
        x = self.inception_module(input, 4, 8, 12, 8, 12, 4)
        x = BatchNormalization()(x)

        # add inception block 2
        x = self.inception_module(x, 8, 16, 24, 16, 24, 8)
        x = BatchNormalization()(x)

        # add inception block 3
        x = self.inception_module(x, 4, 8, 12, 8, 12, 4)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(256, activation=activation)(x)

        return x

    def model4(self, input, activation):
        # add inception block 1
        x = self.inception_module(input, 4, 8, 12, 8, 12, 4)
        x = BatchNormalization()(x)

        x = self.inception_module(x, 4, 8, 12, 8, 12, 4)
        x = BatchNormalization()(x)

        # add inception block 2
        x = self.inception_module(x, 8, 16, 24, 16, 24, 8)
        x = BatchNormalization()(x)

        # add inception block 3
        x = self.inception_module(x, 4, 8, 12, 8, 12, 4)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(128, activation=activation)(x)

        return x

'''
Residual Network
'''
from keras.layers.convolutional import Conv2D
class ResNet(NN):

    def __init__(self, activation, model_number=1):
        # Input layer
        input = Input(shape=(image_height, image_width, 3))

        # --------- Models ---------
        if model_number == 1:
            x = self.model1(input, activation)
        elif model_number == 2:
            x = self.model2(input, activation)
        elif model_number == 3:
            x = self.model3(input, activation)
        elif model_number == 4:
            x = self.model4(input, activation)

        # Output layer
        output = Dense(4, activation='softmax')(x)

        self.model = Model(input, output, name='Inception_v3')

    # function for creating an identity or projection residual module
    def residual_module(self, input, n_filters):
        merge_input = input
        # check if the number of filters needs to be increase, assumes channels last format
        if input.shape[-1] != n_filters:
            merge_input = Conv2D(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(
                input)
        # conv1
        conv1 = Conv2D(n_filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(input)
        # conv2
        conv2 = Conv2D(n_filters, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
        # add filters, assumes filters/channels last
        layer_out = add([conv2, merge_input])
        # activation function
        layer_out = Activation('relu')(layer_out)
        return layer_out

    def model1(self, input, activation):
        x = Conv2D(64, 3, 1, padding="valid", dilation_rate=2)(input)
        x = MaxPooling2D(pool_size=3, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)

        x = self.residual_module(x, 128)
        x = BatchNormalization()(x)

        x = Conv2D(128, 5, 2, padding="valid", dilation_rate=1)(x)
        x = MaxPooling2D(pool_size=5, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)

        x = self.residual_module(x, 64)
        x = BatchNormalization()(x)

        x = Conv2D(256, 3, 1, padding="valid", dilation_rate=1)(x)
        x = AveragePooling2D(pool_size=3, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(256, activation=activation)(x)

        return x

    def model2(self, input, activation):
        x = Conv2D(64, 3, 1, padding="valid", dilation_rate=2)(input)
        x = MaxPooling2D(pool_size=3, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)

        x = self.residual_module(x, 128)
        x = BatchNormalization()(x)

        x = Conv2D(128, 5, 2, padding="valid", dilation_rate=1)(x)
        x = MaxPooling2D(pool_size=5, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)

        x = self.residual_module(x, 64)
        x = BatchNormalization()(x)

        x = Conv2D(32, 3, 1, padding="valid", dilation_rate=2)(x)
        x = AveragePooling2D(pool_size=3, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(128, activation=activation)(x)
        x = Dense(32, activation=activation)(x)

        return x

    def model3(self, input, activation):
        x = Conv2D(64, 3, 1, padding="valid", dilation_rate=2)(input)
        x = BatchNormalization()(x)

        x = self.residual_module(x, 64)
        x = BatchNormalization()(x)

        x = Conv2D(128, 3, 2, padding="valid", dilation_rate=1)(x)
        x = AveragePooling2D(pool_size=3, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)

        x = self.residual_module(x, 64)
        x = BatchNormalization()(x)

        x = Conv2D(64, 3, 1, padding="valid", dilation_rate=1)(x)
        x = AveragePooling2D(pool_size=3, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(64, activation=activation)(x)
        x = Dense(32, activation=activation)(x)

        return x
'''
DNN models
'''
#Model 1

with tf.device('/GPU:0'):
  model1 = DNN('relu', layer=7, batch_norm=True)
  model1.train(X_train, y_train, (X_test, y_test), Nadam(0.001), 100)
  model1.evaluate(X_test, y_test)
  model1.outputResult()
  model1.showPlots()


with tf.device('/GPU:0'):
  model1a = DNN('relu', layer=7, batch_norm=True)
  model1a.train(X_train, y_train, (X_test, y_test), Nadam(0.001), 100)
  model1a.evaluate(X_test, y_test)
  model1a.outputResult()
  model1a.showPlots()


#Model 2

with tf.device('/GPU:0'):
  model2 = DNN('relu', layer=7, batch_norm=True)
  model2.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model2.evaluate(X_test, y_test)
  model2.outputResult()
  model2.showPlots()

with tf.device('/GPU:0'):
  model2a = DNN('relu', layer=7, batch_norm=True)
  model2a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model2a.evaluate(X_test, y_test)
  model2a.outputResult()
  model2a.showPlots()


#Model 3

with tf.device('/GPU:0'):
  model3 = DNN('relu', layer=7, batch_norm=True)
  model3.train(X_train, y_train, (X_test, y_test), Adam(0.001), 100)
  model3.evaluate(X_test, y_test)
  model3.outputResult()
  model3.showPlots()


with tf.device('/GPU:0'):
  model3a = DNN('relu', layer=7, batch_norm=True)
  model3a.train(X_train, y_train, (X_test, y_test), Adam(0.001), 100)
  model3a.evaluate(X_test, y_test)
  model3a.outputResult()
  model3a.showPlots()


#CNN models
#Simple Models
#Model 1

with tf.device('/GPU:0'):
  model4 = CNN('relu', model_number=1)
  model4.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model4.evaluate(X_test, y_test)
  model4.outputResult()
  model4.showPlots()

with tf.device('/GPU:0'):
  model4a = CNN('relu', model_number=1)
  model4a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model4a.evaluate(X_test, y_test)
  model4a.outputResult()
  model4a.showPlots()

#Model 2

with tf.device('/GPU:0'):
  model5 = CNN('relu', model_number=2)
  model5.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model5.evaluate(X_test, y_test)
  model5.outputResult()
  model5.showPlots()

with tf.device('/GPU:0'):
  model5a = CNN('relu', model_number=2)
  model5a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model5a.evaluate(X_test, y_test)
  model5a.outputResult()
  model5a.showPlots()

#Model 3

with tf.device('/GPU:0'):
  model6 = CNN('relu', model_number=3)
  model6.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model6.evaluate(X_test, y_test)
  model6.outputResult()
  model6.showPlots()

with tf.device('/GPU:0'):
  model6a = CNN('relu', model_number=3)
  model6a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model6a.evaluate(X_test, y_test)
  model6a.outputResult()
  model6a.showPlots()


#Dropout Models
#Model 4

with tf.device('/GPU:0'):
  model7 = CNN('relu', model_number=4)
  model7.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model7.evaluate(X_test, y_test)
  model7.outputResult()
  model7.showPlots()
  model7.model.summary()


#Model 5

with tf.device('/GPU:0'):
  model8 = CNN('relu', model_number=5)
  model8.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model8.evaluate(X_test, y_test)
  model8.outputResult()
  model8.showPlots()

#Model 6

with tf.device('/GPU:0'):
  model9 = CNN('relu', model_number=6)
  model9.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model9.evaluate(X_test, y_test)
  model9.outputResult()
  model9.showPlots()

#Pooling Models
#Model 7

with tf.device('/GPU:0'):
  model10 = CNN('relu', model_number=7)
  model10.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model10.evaluate(X_test, y_test)
  model10.outputResult()
  model10.showPlots()

with tf.device('/GPU:0'):
  model10a = CNN('relu', model_number=7)
  model10a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model10a.evaluate(X_test, y_test)
  model10a.outputResult()
  model10a.showPlots()

#Model 8

with tf.device('/GPU:0'):
  model11 = CNN('relu', model_number=8)
  model11.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model11.evaluate(X_test, y_test)
  model11.outputResult()
  model11.showPlots()

with tf.device('/GPU:0'):
  model11a = CNN('relu', model_number=8)
  model11a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model11a.evaluate(X_test, y_test)
  model11a.outputResult()
  model11a.showPlots()

#Model 9

with tf.device('/GPU:0'):
  model12 = CNN('relu', model_number=9)
  model12.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model12.evaluate(X_test, y_test)
  model12.outputResult()
  model12.showPlots()


with tf.device('/GPU:0'):
  model12a = CNN('relu', model_number=9)
  model12a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model12a.evaluate(X_test, y_test)
  model12a.outputResult()
  model12a.showPlots()


#Model 10

with tf.device('/GPU:0'):
  model14 = CNN('relu', model_number=10)
  model14.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model14.evaluate(X_test, y_test)
  model14.outputResult()
  model14.showPlots()

with tf.device('/GPU:0'):
  model14a = CNN('relu', model_number=10)
  model14a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model14a.evaluate(X_test, y_test)
  model14a.outputResult()
  model14a.showPlots()

#Model 11

with tf.device('/GPU:0'):
  model20 = CNN('relu', model_number=11)
  model20.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model20.evaluate(X_test, y_test)
  model20.outputResult()
  model20.showPlots()

with tf.device('/GPU:0'):
  model20a = CNN('relu', model_number=11)
  model20a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model20a.evaluate(X_test, y_test)
  model20a.outputResult()
  model20a.showPlots()

#Model 12
#dropout


with tf.device('/GPU:0'):
  model22 = CNN('relu', model_number=12)
  model22.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model22.evaluate(X_test, y_test)
  model22.outputResult()
  model22.showPlots()


#Inception Net Models
#Model1

with tf.device('/GPU:0'):
  model13 = InceptionNet('relu', model_number=1)
  model13.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model13.evaluate(X_test, y_test)
  model13.outputResult()
  model13.showPlots()
  model13.model.summary()
  plot_model(model13.model, show_shapes=True, to_file='inception_net1.png')

with tf.device('/GPU:0'):
  model13a = InceptionNet('relu', model_number=1)
  model13a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model13a.evaluate(X_test, y_test)
  model13a.outputResult()
  model13a.showPlots()
  model13a.model.summary()
  plot_model(model13a.model, show_shapes=True, to_file='inception_net1.png')

#Model 2

with tf.device('/GPU:0'):
  model14 = InceptionNet('relu', model_number=2)
  model14.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model14.evaluate(X_test, y_test)
  model14.outputResult()
  model14.showPlots()
  model14.model.summary()
  plot_model(model14.model, show_shapes=True, to_file='inception_net2.png')

with tf.device('/GPU:0'):
  model14a = InceptionNet('relu', model_number=2)
  model14a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model14a.evaluate(X_test, y_test)
  model14a.outputResult()
  model14a.showPlots()
  model14a.model.summary()
  plot_model(model14a.model, show_shapes=True, to_file='inception_net2.png')

#Model 3

with tf.device('/GPU:0'):
  model15 = InceptionNet('relu', model_number=3)
  model15.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model15.evaluate(X_test, y_test)
  model15.outputResult()
  model15.showPlots()
  model15.model.summary()
  plot_model(model15.model, show_shapes=True, to_file='inception_net3.png')

with tf.device('/GPU:0'):
  model15a = InceptionNet('relu', model_number=3)
  model15a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model15a.evaluate(X_test, y_test)
  model15a.outputResult()
  model15a.showPlots()
  model15a.model.summary()
  plot_model(model15a.model, show_shapes=True, to_file='inception_net3.png')

#ResNet Models
#Model 1

with tf.device('/GPU:0'):
  model16 = ResNet('relu', model_number=1)
  model16.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model16.evaluate(X_test, y_test)
  model16.outputResult()
  model16.showPlots()
  model16.model.summary()
  plot_model(model16.model, show_shapes=True, to_file='Resnet1.png')

with tf.device('/GPU:0'):
  model16a = ResNet('relu', model_number=1)
  model16a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model16a.evaluate(X_test, y_test)
  model16a.outputResult()
  model16a.showPlots()
  model16a.model.summary()
  plot_model(model16a.model, show_shapes=True, to_file='Resnet1.png')

#Model 2

with tf.device('/GPU:0'):
  model17 = ResNet('relu', model_number=2)
  model17.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model17.evaluate(X_test, y_test)
  model17.outputResult()
  model17.showPlots()

with tf.device('/GPU:0'):
  model17a = ResNet('relu', model_number=2)
  model17a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model17a.evaluate(X_test, y_test)
  model17a.outputResult()
  model17a.showPlots()

#Model 3

with tf.device('/GPU:0'):
    model18 = ResNet('relu', model_number=3)
    model18.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
    model18.evaluate(X_test, y_test)
    model18.outputResult()
    model18.showPlots()

with tf.device('/GPU:0'):
  model18a = ResNet('relu', model_number=3)
  model18a.train(X_train, y_train, (X_test, y_test), Adamax(0.001), 100)
  model18a.evaluate(X_test, y_test)
  model18a.outputResult()
  model18a.showPlots()

#K Fold Cross Validation on Top 3 Models Implementation
def kfoldOnModel(model, inputs, targets, k=5):
    kfold = KFold(n_splits=k, shuffle=True)

    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        copiedModel = deepcopy(model).model

        copiedModel.compile(loss=CategoricalCrossentropy(), optimizer='Adamax', metrics=['accuracy'])

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = copiedModel.fit(inputs[train], targets[train],
                                  batch_size=64,
                                  epochs=100,
                                  verbose=0)

        # predict y_hat for test set
        y_hat_test = copiedModel.predict(inputs[test])
        y_hat_test = argmaxKeepDimensions(y_hat_test)  # convert probablities to 0s and 1s

        # Generate generalization metrics and cofusion matrix
        evaluation_result = copiedModel.evaluate(inputs[test], targets[test], verbose=0)

        print(f'Result for fold {fold_no}: Loss= {evaluation_result[0]}  Acc= {evaluation_result[1] * 100}% \n')
        print("confusion matrix: \n",
              metrics.confusion_matrix(np.argmax(targets[test], axis=1), np.argmax(y_hat_test, axis=1)))
        print(metrics.classification_report(targets[test], y_hat_test, digits=3,
                                            target_names=['Cloudy', 'Rain', 'Shine', 'Sunrise']))

        # Increase fold number
        fold_no = fold_no + 1

#First Top Model
#RestNet
with tf.device('/GPU:0'):
  top1Model = ResNet('relu', model_number=2)

  kfoldOnModel(top1Model, fixed_images, fixed_labels)


#Second Top Model
#CNN with pooling layer
with tf.device('/GPU:0'):
  top2Model = CNN('relu', model_number=11)

  kfoldOnModel(top2Model, fixed_images, fixed_labels)


#Third Top Model
#ResNet
with tf.device('/GPU:0'):
  top3Model = ResNet('relu', model_number=3)

  kfoldOnModel(top3Model, fixed_images, fixed_labels)

#Fourth Top Model
#CNN with pooling


with tf.device('/GPU:0'):
  top4Model = CNN('relu', model_number=7)

  kfoldOnModel(top4Model, fixed_images, fixed_labels)