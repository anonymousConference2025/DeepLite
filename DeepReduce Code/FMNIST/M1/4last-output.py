import keras
from keras import optimizers
from keras.datasets import cifar10,mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from collections import defaultdict
import numpy as np
import sys
import time
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def update_coverage(input_data, model, model_layer_dict, threshold=0.2):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True

def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

# data preprocessing
def data_preprocessing(raw):
    out_y = to_categorical(raw.label, 10)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

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
#print(model1.summary())
#input('check...')
#for layer in model1.layers:
    #for index in range(layer.output_shape[-1]):
    #    print(layer.name)
    #    print(layer.output_shape)
    #print(layer.name)
    #print(layer.output_shape)
    #print(layer.output_shape[-1])
    #print('----------')

model_layer_dict1 = defaultdict(bool)
init_dict(model1,model_layer_dict1)
#print(model_layer_dict1)
#print(len(model_layer_dict1.keys()))
#test_image = x_test[0].reshape([1,32,32,3])
#test_image.shape
#res = model.predict(test_image)
#label = softmax_to_label(res)
#print(label)
#print(x_test[0])
#print(len(x_test[0]))
#print(len(x_test[0][0]))
from keras.models import Model

threshold = float(0.25)
layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
#print(layer_names)
#input('check...')
#intermediate_layer_model = Model(inputs=model1.input,outputs=[model1.get_layer(layer_name).output for layer_name in layer_names])
intermediate_layer_model = Model(inputs=model1.input, outputs = [model1.get_layer(layer_names[-2]).output])

from tqdm import tqdm

cov = []
flag = 0
neuronlist = []

f = open('Cov/cross_entropy','w')

for g in tqdm(range(len(x_train))):
    test_image = x_train[g].reshape([1,28,28,1])

    intermediate_layer_outputs = intermediate_layer_model.predict(test_image)
    #print(type(intermediate_layer_outputs[0]))
    #print(intermediate_layer_outputs[0])
    output = intermediate_layer_outputs[0].tolist()
    #print(output)
    #print(intermediate_layer_output[0])
    #print(len(intermediate_layer_output[0]))
    #input('pause...')
    f.write(str(output) + '\n')
f.close()


# Record the end time
end_time = time.time()
# Calculate the execution time
execution_time = end_time - start_time

# Save the execution time to a file
with open("execution_time_4last_output.txt", "w") as file:
    file.write("Execution time: {} seconds".format(execution_time))

print("Execution time:", execution_time, "seconds")
