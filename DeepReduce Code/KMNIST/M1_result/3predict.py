import keras
from keras import optimizers
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from collections import defaultdict
import numpy as np
import sys
import os
import time
from keras.utils import to_categorical
from pathlib import Path


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


img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

num_classes = 10


if __name__ == '__main__':
    # changelayer = int(sys.argv[1])
    # path = 'doubleneuronnumber-lenet-1/' + str(changelayer) + '/'
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
    model1 = load_model('lenet-1.h5')
    #print(model.summary())
    print(model1.evaluate(x_train,y_train))
    #input('check...')
    count = 0
    acc = 0
    predict = []
    for i in range(len(x_train)):
        test_image = x_train[i].reshape([1,28,28,1])
        y = model1.predict(test_image)
        y_label = np.argmax(y)
        if y_label == np.argmax(y_train[i]):
            acc += 1
            predict.append((1,y_label,np.argmax(y_train[i])))
        else:
            predict.append((0,y_label,np.argmax(y_train[i])))
        count += 1
        #print('%s - %s'%(y,np.argmax(y_test[i])))
    # print(model1.evaluate(x_test,y_test))
    print("total : %s, acc : %s, accratio : %s"%(count,acc,acc/(count*1.0)))
    f = open('Cov/predict','w')
    for i in range(len(x_train)):
        f.write(str(predict[i]) + '\n')
    f.close()

    # Record the end time
    end_time = time.time()
    # Calculate the execution time
    execution_time = end_time - start_time

    # Save the execution time to a file
    with open("execution_time_3predict.txt", "w") as file:
        file.write("Execution time: {} seconds".format(execution_time))

    print("Execution time:", execution_time, "seconds")
    
