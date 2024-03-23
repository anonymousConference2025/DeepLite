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
from keras.utils import to_categorical
from pathlib import Path
import time
from keras.optimizers import SGD


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

batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    sgd = SGD(learning_rate=0.01, momentum=0.9) # Example custom parameters
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # Record the end time
    start_time = time.time()
    # Let us define some paths first
    input_path = Path("./input")

    # Path to training images and corresponding labels provided as numpy arrays
    kmnist_train_images_path = input_path/"kmnist-train-imgs.npz"
    kmnist_train_labels_path = input_path/"kmnist-train-labels.npz"

    # Path to the test images and corresponding labels
    kmnist_test_images_path = input_path/"kmnist-test-imgs.npz"
    kmnist_test_labels_path = input_path/"kmnist-test-labels.npz"

    # Load the training data from the corresponding npz files
    kmnist_train_images = np.load(kmnist_train_images_path)['arr_0']
    kmnist_train_labels = np.load(kmnist_train_labels_path)['arr_0']

    # Load the test data from the corresponding npz files
    kmnist_test_images = np.load(kmnist_test_images_path)['arr_0']
    kmnist_test_labels = np.load(kmnist_test_labels_path)['arr_0']

    print(f"Number of training samples: {len(kmnist_train_images)} where each sample is of size: {kmnist_train_images.shape[1:]}")
    print(f"Number of test samples: {len(kmnist_test_images)} where each sample is of size: {kmnist_test_images.shape[1:]}")


    # Process the train and test data in the exact same manner as done for MNIST
    x_train = kmnist_train_images.astype('float32')
    x_test = kmnist_test_images.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)

    # convert class vectors to binary class matrices
    y_train = to_categorical(kmnist_train_labels, num_classes)
    y_test = to_categorical(kmnist_test_labels, num_classes)

    print(x_train.shape)

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

