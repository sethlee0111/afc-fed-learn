import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import tensorflow.keras as keras

SIZE_X = 28
SIZE_Y = 28

# get X and Y data from a list of files
# returns: list of numpy arrays (num_samples_from_user, num_pixels)
def get_data(filenames):
    i = 0
    X = []
    Y = []
    users = []
    for fn in filenames:
        i += 1
        print("\r({}/{}) processing: {}".format(i, len(filenames), fn))
        with open(fn, "r") as f:
            data = f.read()
        parsed_data = json.loads(data)
        X.extend([np.array(parsed_data['user_data'][user]['x']) for user in parsed_data['users']])
        Y.extend([np.array(parsed_data['user_data'][user]['y']) for user in parsed_data['users']])
        users.extend(parsed_data['users'])
    return X, Y, users

# visualize the handwritten letters 
def visualize_writings(writing):
    map2d = []
    for i in range(0, len(writing), SIZE_Y):
        map2d.append(writing[i:i+SIZE_X])
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.array(map2d))
    fig.tight_layout()
    plt.show()
    
# def preprocess(X):
    
    
# parse data to global dataset and local sets for federated settings 
# every user is allocated to a global or local dataset, not being shared by two different sets
# this function only tries its best to fulfill requirements, it doesn't do error checking
# args: 
#      X: list of numpy arrays (num_samples_from_user, num_pixels)
#      num_global: minimum number of global data
#      num_local: minimum number of local data
# returns: X, Y for global, list of (X, Y)s for locals
def fl_parse(X, Y, num_clients, min_num_global, min_num_local):
    X_global = []
    Y_global = []
    local_data = []
    cnt = 0
    i = 0
    while i < len(X):
        X_global.append(X[i])
        Y_global.append(Y[i])
        cnt += X[i].shape[0]
        i += 1
        if cnt > min_num_global:
            break
            
    while len(local_data) < num_clients and i < len(X):
        X_local = []
        Y_local = []
        cnt = 0
        while cnt < min_num_local:
            X_local.append(X[i])
            Y_local.append(Y[i])
            cnt += X[i].shape[0]
            i += 1
        local_data.append((serialize_data(X_local), serialize_data(Y_local)))
        
    return serialize_data(X_global), serialize_data(Y_global), local_data

# split training set with given size and number
# args:
#      size: number of the data in each training set
#      x_train: numpy array of shape (num_samples, num_dimensions)
# returns: list of numpy array for X, Y
def split_training_set(size, number, x_train, y_train):
    x_train_list = np.split(x_train, x_train.shape[0] / size)[:number]  # +1 cuz the last array will contain everything till the end
    y_train_list = np.split(y_train, y_train.shape[0] / size)[:number]
    y_train_list = [keras.utils.to_categorical(y, len(np.unique(y_train))) for y in y_train_list]
    return x_train_list, y_train_list

def split_training_set_by_number(number, x_train, y_train):
    x_train_list = np.split(x_train, number)

def split_training_set_unbalanced(start_size, diff, number, x_train, y_train):
    num_shards = int(number * (number + 1) / 2)
    x_train_shards = np.split(x_train, x_train.shape[0] / diff)[:num_shards]
    y_train_shards = np.split(y_train, y_train.shape[0] / diff)[:num_shards]
    
    x_train_list = []
    y_train_list = []
    for i in range(number):
        if len(x_train_shards[:i+1]) != i+1:
            raise ValueError('train dataset not enough to construct given number of training set')
        x_train_list.append(np.concatenate(x_train_shards[:i+1], axis=0))
        x_train_shards = x_train_shards[i+1:]
        y_train_list.append(np.concatenate(y_train_shards[:i+1], axis=0))
        y_train_shards = y_train_shards[i+1:]
        
    y_train_list = [keras.utils.to_categorical(y, len(np.unique(y_train))) for y in y_train_list]
    return x_train_list, y_train_list

def filter_data_by_labels(x_train, y_train, labels, size=-1, noise=0, randseed=0):
    """
    return only the data with corresponding labels with noise
    note that the resulting size could be different from the parameter size.
    This is to ensure the number of data for each labels are exactly equal 
    """
    np.random.seed(randseed)
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]
    
    data_size = len(y_train)
    
    if size != -1:
        data_per_label = (int)(size / len(labels))
    
    mask = np.zeros(y_train.shape, dtype=bool)
    
    for l in labels:
        new_mask = (y_train == l)
        cnt = 0
        for i in range(data_size):
            if new_mask[i]:
                cnt += 1
                if cnt >= data_per_label:
                    break

        mask |= np.append(new_mask[:i+1], np.zeros(data_size-i-1, dtype=bool))
    
    noise_mask = np.logical_not(mask)
       
    if size > 0 and size <= data_size:
        num_noise = (int)(size * noise)
        return np.concatenate((x_train[noise_mask][:num_noise], x_train[mask][:size-num_noise]), axis=0),\
               np.concatenate((y_train[noise_mask][:num_noise], y_train[mask][:size-num_noise]), axis=0)
    else:
        return x_train[mask], y_train[mask]
    
def filter_data_by_labels_with_numbers(x_train, y_train, labels, nums, randseed=0):
    """
    nums: a dict that specifies the number of data points per labels
    """
    if type(nums) != type({}):
        raise TypeError("nums has to be a dict type, not {}".format(type(nums)))
    
    np.random.seed(randseed)
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]
    
    data_size = len(y_train)
    
    mask = np.zeros(y_train.shape, dtype=bool)
    
    for l in labels:
        new_mask = (y_train == l)
        cnt = 0
        for i in range(data_size):
            if cnt >= nums[l]:
                break
            if new_mask[i]:
                cnt += 1

        mask |= np.append(new_mask[:i+1], np.zeros(data_size-i-1, dtype=bool))
        
    return x_train[mask], y_train[mask]
    
# change list of numpy arrays (num_samples_from_user, num_pixels) to
# list of numpy arrays (num_pixels)
# in other words, erase user info and just serialize all the data
def serialize_data(X):
    res = []
    for x in X:
        res.extend(list(x))
    return np.array(res)

def get_train_data_from_filename(n):
    return "all_data_" + str(n) + "_niid_0_keep_10_train_9.json"

def get_test_data_from_filename(n):
    return "all_data_" + str(n) + "_niid_0_keep_10_test_9.json"
