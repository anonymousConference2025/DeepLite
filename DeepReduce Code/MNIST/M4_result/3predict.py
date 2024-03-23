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


def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test



if __name__ == '__main__':
    # changelayer = int(sys.argv[1])
    # path = 'doubleneuronnumber-lenet-1/' + str(changelayer) + '/'
    # Record the end time
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 处理 x
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_test = x_test.astype('float32')

    x_test /= 255

    # 处理 y
    y_train = keras.utils.to_categorical(y_train)

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
    
