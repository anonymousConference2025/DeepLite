#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.optimizers import Adam




import keras
import numpy as np
from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization
from keras.initializers import RandomNormal  
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard

from keras import initializers
from keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python import keras
from keras.utils import to_categorical



import time


from keras import backend as K
if 'tensorflow' == K.backend():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate memory on GPU 0
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

        # Allow memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

batch_size = 256
num_classes = 10
epochs = 10

# img_rows, img_cols = 28, 28
# input_shape = (img_rows, img_cols, 1)

def build_model():
    model = Sequential()

    model.add(Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_normal', input_shape=(28,28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same', activation='selu'))
    model.add(Conv2D(128, 3, padding='same', activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation='selu'))

    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    # opt = Adam(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


# data preprocessing
def data_preprocessing(raw):
    out_y = to_categorical(raw.label, 10)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


if __name__ == '__main__':
    # Record the end time
    start_time = time.time()
    train_data = pd.read_csv('./fmnist/fashion-mnist_train.csv')
    test_data = pd.read_csv('./fmnist/fashion-mnist_test.csv')
    print("Fashion MNIST train -  rows:",train_data.shape[0]," columns:", train_data.shape[1])
    print("Fashion MNIST test -  rows:",test_data.shape[0]," columns:", test_data.shape[1])
        # build network

    # prepare the data
    X, y = data_preprocessing(train_data)
    x_test, y_test = data_preprocessing(test_data)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)

    model = build_model()
    model.summary()
    history = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_val, y_val))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    # save model
    # path = 'Cov/model/'
    file_name = 'lenet-1.h5'
    model.save(file_name)
    # Record the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time
    # Save the execution time to a file
    with open("execution_time_lenet.txt", "w") as file:
        file.write("Execution time: {} seconds".format(execution_time))

    print("Execution time:", execution_time, "seconds")

