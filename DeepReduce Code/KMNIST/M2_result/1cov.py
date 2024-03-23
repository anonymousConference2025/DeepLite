import keras
from keras import optimizers
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import sys
import os
import argparse
from keras.utils import to_categorical
from pathlib import Path
import time





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
    # 为true的有多少
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


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



'''
# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
                help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
                help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
                help='dataset. (default: cifar10)')

args = parser.parse_args()

stack_n            = args.stack_n
layers             = 6 * stack_n + 2
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = args.batch_size
epochs             = args.epochs
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4

#from tqdm import tqdm
'''

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10


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
print("test...")
score = model1.evaluate(x_train, y_train, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
# input('check,,,')
#for layer in model1.layers:
    #for index in range(layer.output_shape[-1]):
    #    print(layer.name)
    #    print(layer.output_shape)
    #print(layer.name)
    #print(layer.output_shape)
    #print(layer.output_shape[-1])
    #print('----------')

model_layer_dict1 = defaultdict(bool)
init_dict(model1, model_layer_dict1)
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

#threshold = float(0.5)
threshold = float(sys.argv[1])
print(threshold)
layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
intermediate_layer_model = Model(inputs=model1.input,outputs=[model1.get_layer(layer_name).output for layer_name in layer_names])

cov = []
flag = 0
neuronlist = []

# for g in tqdm(range(len(x_test))):
for g in tqdm(range(len(x_train))):
    # test_image = x_test[g].reshape([1,28,28,1])
    test_image = x_train[g].reshape([1,28,28,1])
    #print(model1.predict(test_image))
    #print(y_test[g])
    #print('*****************')
    #intermediate_layer_model = Model(inputs=model.input,outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(test_image)
    #print(intermediate_layer_outputs)
    tempcount = 0
    tempstr = ''
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        # print("type:",type(intermediate_layer_output))      # type: <class 'numpy.ndarray'>
        # print("shape:",intermediate_layer_output.shape)     # (1,*)
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            #if np.mean(scaled[..., num_neuron]) > threshold:
            if np.max(scaled[..., num_neuron]) > threshold:
                tempcount += 1
                tempstr += '1'
                if model_layer_dict1[(layer_names[i], num_neuron)] == False:
                    model_layer_dict1[(layer_names[i], num_neuron)] = True
                #    print("%s, %s : %s"%(layer_names[i], num_neuron,model_layer_dict1[(layer_names[i], num_neuron)]))
            else:
                tempstr += '0'
            if flag == 1:
                continue
            else:
                neuronlist.append((layer_names[i], num_neuron))
    cov.append(tempstr)
    flag = 1

    #print('%d : %d '%(g+1,tempcount))
    #print('*****************')

tempcount = 0
totalcount = 0
for key in model_layer_dict1:
    totalcount += 1
    if model_layer_dict1[key] == True:
        tempcount += 1
print(model_layer_dict1)
print('%d / %d'%(tempcount,totalcount))

if os.path.exists('Cov/activeneuron/'+str(threshold)+'ase/') == False:
    os.makedirs('Cov/activeneuron/'+str(threshold)+'ase/')

f = open('Cov/activeneuron/'+str(threshold)+'ase/neuron_cov','w')
# t = open(path+'Cov/activeneuron/'+str(threshold)+'ase/testinput','w')
for i in range(len(cov)):
    f.write(cov[i] + '\n')
    # t.write(str(x_test[i]) + '\n')
f.close()
# t.close()

n = open('Cov/activeneuron/'+str(threshold)+'ase/neuron','w')
for neuron in neuronlist:
    n.write(str(neuron) + '\n')
n.close()


# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

# Save the execution time to a file
with open("execution_time_1con.txt", "w") as file:
    file.write("Execution time: {} seconds".format(execution_time))

print("Execution time:", execution_time, "seconds")
