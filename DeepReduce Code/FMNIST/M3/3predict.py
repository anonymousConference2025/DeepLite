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
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



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


# data preprocessing
def data_preprocessing(raw):
    out_y = to_categorical(raw.label, 10)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y



if __name__ == '__main__':
    # changelayer = int(sys.argv[1])
    # path = 'doubleneuronnumber-lenet-1/' + str(changelayer) + '/'
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
    
