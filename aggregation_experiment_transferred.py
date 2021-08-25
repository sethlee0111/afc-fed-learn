from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import tensorflow as tf
import utils
import semantic_drift

# Fully federated, one-to-one model from the initial model

# Hyperparameters
batch_size = 50
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28


def custom_model(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_model(model):  
    # initiate SGD optimizer
    opt = keras.optimizers.SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    
def fit_model_with_datasets(model, epochs, x_train, y_train):
    now = datetime.datetime.now()
#     print ("Training date and time : ")
#     print (now.strftime("%Y-%m-%d %H:%M:%S"))
    res =  model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      shuffle=True,
                      verbose=0)
#     print ("Elasped Time: " + str(datetime.datetime.now() - now))
    return res

def model_combs(model_list):
    combs = list()
    l = len(model_list)
    for i in range(l):
        for j in range(l):
            if i > j:
                combs.append([model_list[i], model_list[j]])
    return combs

def run(seed):
    print("seed {}".format(seed))
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    np.random.seed(seed)
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    global_dataset_size = 0
    local_dataset_size = 60000

    X_global = x_train[-global_dataset_size:]
    Y_global = y_train[-global_dataset_size:]
    X_local = x_train[:-global_dataset_size]
    Y_local = y_train[:-global_dataset_size]

    X_local_list, Y_local_list = utils.split_training_set(3000, 20, X_local, Y_local)

    # convert class vectors to binary class matrices
    num_classes = 10
    Y_global = keras.utils.to_categorical(Y_global, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model1 = custom_model(input_shape, num_classes)
    compile_model(model1)
    fit_model_with_datasets(model1, 1, X_global, Y_global)

    model_list = list()
    for _ in range(20):
        model_list.append(tf.keras.models.clone_model(model1)) 
        model_list[_].set_weights(model1.get_weights())

    # sort models according to similarity. We arbitrarily take the model1 as a "standard"
    standard_model = tf.keras.models.clone_model(model1)
    standard_model.set_weights(model_list[0].get_weights())

    for i in range(len(model_list)):
        compile_model(model_list[i])
        fit_model_with_datasets(model_list[i], (i+1)*10, X_local_list[i], Y_local_list[i])

    model_list.sort(key=lambda m : semantic_drift.l2_distance(standard_model, m))

    theta_list = [0, 0.5, 1]
    agg_weights_list_per_pi = list()
    dist_list = list()

    for model_comp in model_combs(model_list):
        if model_comp[0] is model_comp[1]:    #disregard same models
            continue
        weights = [model_comp[0].get_weights(), model_comp[1].get_weights()]
        agg_weights_list = list()
        for theta in theta_list:
            agg_weights = list()
            for weights_list_tuple in zip(*weights):
                agg_weights.append(np.array([np.average(np.array(w), axis=0, weights=[1. - theta, theta]) for w in zip(*weights_list_tuple)]))
            agg_weights_list.append(agg_weights)
        dist_list.append(semantic_drift.l2_distance(model_comp[0], model_comp[1]))
        agg_weights_list_per_pi.append(agg_weights_list)

        agg_weights_list_per_pi_sorted = [x for _,x in sorted(zip(dist_list,agg_weights_list_per_pi))]
        model_combs_sorted = [x for _,x in sorted(zip(dist_list, model_combs(model_list)))]

    B = np.zeros(len(agg_weights_list_per_pi))

    i = 0
    for agg_weights_list in agg_weights_list_per_pi_sorted:

        aggr_model = keras.models.clone_model(model1)
        aggr_model.set_weights(agg_weights_list[1])
        compile_model(aggr_model)
        score = aggr_model.evaluate(x=x_test, y=y_test, verbose=0)
        
        aggr_model = keras.models.clone_model(model1)
        aggr_model.set_weights(agg_weights_list[0])
        compile_model(aggr_model)
        comp_score1 = aggr_model.evaluate(x=x_test, y=y_test, verbose=0)
        
        aggr_model = keras.models.clone_model(model1)
        aggr_model.set_weights(agg_weights_list[2])
        compile_model(aggr_model)
        comp_score2 = aggr_model.evaluate(x=x_test, y=y_test, verbose=0)
        
        B[i] = score[0] - min(comp_score1[0], comp_score2[0])
        K.clear_session() #prevent memory leak https://github.com/keras-team/keras/issues/13118
        i += 1
        if i % 10 == 0:
            print("{}th iteration".format(i))

    return B, dist_list