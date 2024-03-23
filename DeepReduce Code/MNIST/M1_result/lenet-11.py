import keras
import numpy as np
from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import RandomNormal  
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras import initializers
from keras import regularizers


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
epochs = 90

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=6, input_shape=(28,28,1), kernel_size=(5, 5),strides=(1, 1), padding="valid", data_format='channels_last',  kernel_initializer = initializers.VarianceScaling( scale=2.0, mode='fan_in', distribution='normal'), kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=1e-4 )))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format='channels_last'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),strides=(1, 1), padding="valid", data_format='channels_last',  kernel_initializer = initializers.VarianceScaling( scale=2.0, mode='fan_in', distribution='normal'), kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=1e-4 )))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format='channels_last'))
    model.add(Flatten(data_format = 'channels_last'))
    model.add(Dense(units=120,  kernel_initializer = initializers.VarianceScaling( scale=2.0, mode='fan_in', distribution='normal'), kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=1e-4 )))
    model.add(Activation('relu'))
    model.add(Dense(units=84,  kernel_initializer = initializers.VarianceScaling( scale=2.0, mode='fan_in', distribution='normal'), kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=1e-4 )))
    model.add(Activation('relu'))
    model.add(Dense(units=10, kernel_initializer = initializers.VarianceScaling( scale=2.0, mode='fan_in', distribution='normal'), kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=1e-4 )))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # Record the end time
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 处理 x
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    # 处理 y
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # build network
    model = build_model()
    print(model.summary())

    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)

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

