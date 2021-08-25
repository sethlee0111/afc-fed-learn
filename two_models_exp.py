from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import semantic_drift
import tensorflow as tf
import utils

def get_result_filename(params):
    now = datetime.datetime.now()
    res_filename = "{}".format(now.strftime("%H:%M:%S"))
    res_filename += "epochs-" + str(params['epochs'])+ '_'
    res_filename += "size-" + str(params['size']) + '_'

    return res_filename

def custom_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def custom_model_cifar():
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def compile_model(model):  
    # initiate SGD optimizer
    opt = keras.optimizers.SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

def compile_model_lr(model, lr):  
    # initiate SGD optimizer
    opt = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

def fit_model_with_datasets(model, epochs, x_train, y_train):
    now = datetime.datetime.now()
    return model.fit(x_train, y_train,
                      batch_size=50,
                      epochs=epochs,
                      shuffle=True, verbose=0)

def fit_model_with_datasets_and_val_set(model, epochs, x_train, y_train, val_set):
    now = datetime.datetime.now()
    return model.fit(x_train, y_train,
                      batch_size=50,
                      validation_data=val_set,
                      epochs=epochs,
                      shuffle=True, verbose=1)

def add_hist_to_dict(d, hist):
    if 'loss' not in d:
        d['loss'] = hist[0]
    else:
        d['loss'] = np.append(d['loss'], hist[0])
    if 'acc' not in d:
        d['acc'] = hist[1]
    else:
        d['acc'] = np.append(d['acc'], hist[1])

def get_losses_for_overlapping_labels_sliding_window(model, epochs, x_train, y_train, x_test, y_test, size, diff):
    """
    experiment for "sliding window" approach
    """
    num_total_classes = 10 # todo np.unique(y_train)
    
    res = {}
    res['model_aggr'] = {}
    res['model_0to4'] = {}
    res['model_5to9'] = {}
    res['model_aggr']['test_all'] = {}
    res['model_aggr']['test_0to4'] = {}
    res['model_aggr']['test_5to9'] = {}
    res['model_aggr']['test_0'] = {}
    
    
    res['model_0to4'] = copy.deepcopy(res['model_aggr'])
    res['model_5to9'] = copy.deepcopy(res['model_aggr'])
    
    additionals = {}
    additionals['l2_dist'] = np.array([])
    additionals['req_rounds'] = np.array([])
    
    y_test_one_hot = keras.utils.to_categorical(y_test, num_total_classes)
    tx1, ty1 = utils.filter_data_by_labels(x_test, y_test, np.arange(5), 1000)
    tx2, ty2 = utils.filter_data_by_labels(x_test, y_test, np.arange(5)+5, 1000)
    tx0, ty0 = utils.filter_data_by_labels(x_test, y_test, np.arange(1), 200)
    
    ty1 = keras.utils.to_categorical(ty1, num_total_classes)
    ty2 = keras.utils.to_categorical(ty2, num_total_classes)
    ty0 = keras.utils.to_categorical(ty0, num_total_classes)
    
    label_move = 2
    
    for r in np.arange(0, label_move, diff):
        print("--------------  Iteration #{}  --------------".format((int)(r/diff)))
        
        # get data
        randseed = (int)(datetime.datetime.now().microsecond)
        x1, y1 = utils.filter_data_by_labels(x_train, y_train, np.arange(5), size, 0, randseed)
        
        label_conf = {}
        start_label = (int)(r)
        end_label = 4 + start_label
        for i in np.arange(0, start_label, 1):
            label_conf[i] = 0
        for i in np.arange(start_label, end_label+2, 1):
            if i == start_label:
                label_conf[i] = (int)(120 * (1-(r-start_label)))
            elif i == end_label+1:
                label_conf[i] = 120 - (label_conf[start_label])
            else:
                label_conf[i] = 120

        x2, y2 = utils.filter_data_by_labels_with_numbers(x_train, 
                                                          y_train, 
                                                          label_conf,
                                                          randseed+1000)

        y1 = keras.utils.to_categorical(y1, num_total_classes)
        y2 = keras.utils.to_categorical(y2, num_total_classes)
        
        # initialize models
        model1 = keras.models.clone_model(model)
        model2 = keras.models.clone_model(model)
        model1.set_weights(model.get_weights())
        model2.set_weights(model.get_weights())
        compile_model(model1)
        compile_model(model2)
        
        # fit
        fit_model_with_datasets(model1, 20, x1, y1)
        fit_model_with_datasets(model2, 20, x2, y2)
        
        additionals['l2_dist'] = np.append(additionals['l2_dist'], np.array([semantic_drift.l2_distance(model1, model2)]))

        pre_eval_res = model1.evaluate(x=tx1, y=ty1, verbose=0)
        
        # test
        # add_hist_to_dict(res['model_0to4']['test_all'],
        #                  model1.evaluate(x=x_test, y=y_test_one_hot, verbose=0))
        add_hist_to_dict(res['model_0to4']['test_0to4'],
                         pre_eval_res)
        # add_hist_to_dict(res['model_0to4']['test_5to9'],
        #                  model1.evaluate(x=tx2, y=ty2, verbose=0))
        # add_hist_to_dict(res['model_0to4']['test_0'],
        #                  model1.evaluate(x=tx0, y=ty0, verbose=0))
        
        # add_hist_to_dict(res['model_5to9']['test_all'],
        #                  model2.evaluate(x=x_test, y=y_test_one_hot, verbose=0))
        add_hist_to_dict(res['model_5to9']['test_0to4'],
                         model2.evaluate(x=tx1, y=ty1, verbose=0))
        # add_hist_to_dict(res['model_5to9']['test_5to9'],
        #                  model2.evaluate(x=tx2, y=ty2, verbose=0))
        # add_hist_to_dict(res['model_5to9']['test_0'],
        #                  model2.evaluate(x=tx0, y=ty0, verbose=0))
        

        rounds = 100
        rounds_max = 15
        for ro in range(rounds):
            #aggregate
            weights = [model1.get_weights(), model2.get_weights()]
            agg_weights = list()
            theta = 0.5
            for weights_list_tuple in zip(*weights):
                agg_weights.append(np.array([np.average(np.array(w), axis=0, weights=[1. - theta, theta]) for w in zip(*weights_list_tuple)]))
            aggr_model = keras.models.clone_model(model1)
            aggr_model.set_weights(agg_weights)
            compile_model(aggr_model)
            
            model1.set_weights(agg_weights)
            model2.set_weights(agg_weights)
            
            # stop criteria
            eval_res = aggr_model.evaluate(x=tx1, y=ty1, verbose=0)
            if eval_res[1] >= pre_eval_res[1] + 0.02:
                additionals['req_rounds'] = np.append(additionals['req_rounds'], ro)
                break;
            
            # exceeded max rounds
            if ro > rounds_max:
                additionals['req_rounds'] = np.append(additionals['req_rounds'], ro)
                break;
            
            # fit
            if ro < rounds:
                print("o", end='')
                fit_model_with_datasets(model1, epochs, x1, y1)
                fit_model_with_datasets(model2, epochs, x2, y2)

        print("")
        
        # test
        # add_hist_to_dict(res['model_aggr']['test_all'],
        #                  aggr_model.evaluate(x=x_test, y=y_test_one_hot, verbose=0))
        add_hist_to_dict(res['model_aggr']['test_0to4'],
                         eval_res)
        # add_hist_to_dict(res['model_aggr']['test_5to9'],
        #                  aggr_model.evaluate(x=tx2, y=ty2, verbose=0))
        # add_hist_to_dict(res['model_aggr']['test_0'],
        #                  aggr_model.evaluate(x=tx0, y=ty0, verbose=0))
        
    return res, additionals

def get_losses_for_overlapping_labels(model, epochs, x_train, y_train, x_test, y_test, size, diff):
    """
    experiment for "sliding window" approach, but data is subbed by the noise, not the next label
    """
    num_total_classes = 10 # todo np.unique(y_train)
    
    res = {}
    res['model_aggr'] = {}
    res['model_0to4'] = {}
    res['model_5to9'] = {}
    res['model_aggr']['test_all'] = {}
    res['model_aggr']['test_0to4'] = {}
    res['model_aggr']['test_5to9'] = {}
    res['model_aggr']['test_0'] = {}
    
    
    res['model_0to4'] = copy.deepcopy(res['model_aggr'])
    res['model_5to9'] = copy.deepcopy(res['model_aggr'])
    
    additionals = {}
    additionals['l2_dist'] = np.array([])
    additionals['req_rounds'] = np.array([])
    
    y_test_one_hot = keras.utils.to_categorical(y_test, num_total_classes)
    tx1, ty1 = utils.filter_data_by_labels(x_test, y_test, np.arange(5), 1000)
    tx2, ty2 = utils.filter_data_by_labels(x_test, y_test, np.arange(5)+5, 1000)
    tx0, ty0 = utils.filter_data_by_labels(x_test, y_test, np.arange(1), 200)
    
    ty1 = keras.utils.to_categorical(ty1, num_total_classes)
    ty2 = keras.utils.to_categorical(ty2, num_total_classes)
    ty0 = keras.utils.to_categorical(ty0, num_total_classes)
    
    for r in np.arange(0, 1 + diff, diff):
        print("--------------  Iteration #{}  --------------".format((int)(r/diff)))
        
        # get data
        randseed = (int)(datetime.datetime.now().microsecond)
        x1, y1 = utils.filter_data_by_labels(x_train, y_train, np.arange(5), size, 0, randseed)
        
        label_conf = {}
        start_label = (int)(r)
        end_label = 4 + start_label
        for i in np.arange(0, start_label, 1):
            label_conf[i] = 0
        for i in np.arange(start_label, end_label+2, 1):
            if i == start_label:
                label_conf[i] = (int)(120 * (1-(r-start_label)))
            elif i == end_label+1:
                label_conf[i] = 120 - (label_conf[start_label])
            else:
                label_conf[i] = 120

        x2, y2 = utils.filter_data_by_labels(x_train, 
                                             y_train, 
                                             np.arange(5), 
                                             size,
                                             r,
                                             randseed+1000)

        y1 = keras.utils.to_categorical(y1, num_total_classes)
        y2 = keras.utils.to_categorical(y2, num_total_classes)
        
        # initialize models
        model1 = keras.models.clone_model(model)
        model2 = keras.models.clone_model(model)
        model1.set_weights(model.get_weights())
        model2.set_weights(model.get_weights())
        compile_model(model1)
        compile_model(model2)
        
        # fit
        fit_model_with_datasets(model1, 20, x1, y1)
        fit_model_with_datasets(model2, 20, x2, y2)
        
        additionals['l2_dist'] = np.append(additionals['l2_dist'], np.array([semantic_drift.l2_distance(model1, model2)]))

        pre_eval_res = model1.evaluate(x=tx1, y=ty1, verbose=0)
        
        # test
        # add_hist_to_dict(res['model_0to4']['test_all'],
        #                  model1.evaluate(x=x_test, y=y_test_one_hot, verbose=0))
        add_hist_to_dict(res['model_0to4']['test_0to4'],
                         pre_eval_res)
        # add_hist_to_dict(res['model_0to4']['test_5to9'],
        #                  model1.evaluate(x=tx2, y=ty2, verbose=0))
        # add_hist_to_dict(res['model_0to4']['test_0'],
        #                  model1.evaluate(x=tx0, y=ty0, verbose=0))
        
        # add_hist_to_dict(res['model_5to9']['test_all'],
        #                  model2.evaluate(x=x_test, y=y_test_one_hot, verbose=0))
        add_hist_to_dict(res['model_5to9']['test_0to4'],
                         model2.evaluate(x=tx1, y=ty1, verbose=0))
        # add_hist_to_dict(res['model_5to9']['test_5to9'],
        #                  model2.evaluate(x=tx2, y=ty2, verbose=0))
        # add_hist_to_dict(res['model_5to9']['test_0'],
        #                  model2.evaluate(x=tx0, y=ty0, verbose=0))
        

        rounds = 100
        rounds_max = 15
        for ro in range(rounds):
            #aggregate
            weights = [model1.get_weights(), model2.get_weights()]
            agg_weights = list()
            theta = 0.5
            for weights_list_tuple in zip(*weights):
                agg_weights.append(np.array([np.average(np.array(w), axis=0, weights=[1. - theta, theta]) for w in zip(*weights_list_tuple)]))
            aggr_model = keras.models.clone_model(model1)
            aggr_model.set_weights(agg_weights)
            compile_model(aggr_model)
            
            model1.set_weights(agg_weights)
            model2.set_weights(agg_weights)
            
            # stop criteria
            eval_res = aggr_model.evaluate(x=tx1, y=ty1, verbose=0)
            if eval_res[1] >= pre_eval_res[1] + 0.02:
                additionals['req_rounds'] = np.append(additionals['req_rounds'], ro)
                break;
            
            # exceeded max rounds
            if ro > rounds_max:
                additionals['req_rounds'] = np.append(additionals['req_rounds'], ro)
                break;
            
            # fit
            if ro < rounds:
                print("o", end='')
                fit_model_with_datasets(model1, epochs, x1, y1)
                fit_model_with_datasets(model2, epochs, x2, y2)

        print("")
        
        # test
        # add_hist_to_dict(res['model_aggr']['test_all'],
        #                  aggr_model.evaluate(x=x_test, y=y_test_one_hot, verbose=0))
        add_hist_to_dict(res['model_aggr']['test_0to4'],
                         eval_res)
        # add_hist_to_dict(res['model_aggr']['test_5to9'],
        #                  aggr_model.evaluate(x=tx2, y=ty2, verbose=0))
        # add_hist_to_dict(res['model_aggr']['test_0'],
        #                  aggr_model.evaluate(x=tx0, y=ty0, verbose=0))
        
    return res, additionals

def get_losses_for_overlapping_labels_w_noise(get_model_func, 
                                              pretrained_weights,
                                              local_epochs, 
                                              epochs,
                                              max_rounds, 
                                              x_train, 
                                              y_train, 
                                              x_test, 
                                              y_test, 
                                              size, 
                                              fr_data_size,
                                              diff):
    """
    experiment for simply adding noise approach
    diff: step size of the percentage of labels 0-4 substituted with 5-9
    """
    num_total_classes = 10 # todo np.unique(y_train)
    target_labels = np.array([0, 1])
    
    res = {}
    res['model_aggr'] = {}
    res['model_0to4'] = {}
    res['model_5to9'] = {}
    res['model_aggr']['test_all'] = {}
    res['model_aggr']['test_0to4'] = {}
    res['model_aggr']['test_5to9'] = {}
    res['model_aggr']['test_0'] = {}
    
    
    res['model_0to4'] = copy.deepcopy(res['model_aggr'])
    res['model_5to9'] = copy.deepcopy(res['model_aggr'])
    
    additionals = {}
    additionals['l2_dist'] = np.array([])
    additionals['req_rounds'] = np.array([])
    
    y_test_one_hot = keras.utils.to_categorical(y_test, num_total_classes)
    tx1, ty1 = utils.filter_data_by_labels(x_test, y_test, target_labels, 1000)
    
    ty1 = keras.utils.to_categorical(ty1, num_total_classes)

    additionals['loss_benefit'] = {}
    for i in range(max_rounds):
        additionals['loss_benefit'][i] = []
    
    for r in np.arange(0, 1 + diff, diff):
        print("--------------  Iteration #{}  --------------".format((int)(r/diff)+1))
        
        # get data
        randseed = (int)(datetime.datetime.now().microsecond)
        np.random.seed(randseed)
        x1, y1 = utils.filter_data_by_labels(x_train, y_train, target_labels, size, 0)
        
        label_conf = {}

        noise_labels = np.setdiff1d(np.arange(10), target_labels)
        np.random.shuffle(noise_labels)
        noise_labels = noise_labels[:2]
        noise_labels = np.arange(3,5)

        # @TODO the resulting data size is not always [size]
        noise_size_per_label = (int)(size * r / len(noise_labels))
        target_label_size = (size - noise_size_per_label * len(noise_labels)) / len(target_labels)

        for i in target_labels:
            label_conf[i] = target_label_size
        for i in noise_labels:
            label_conf[i] = noise_size_per_label

        print(label_conf)

        x2, y2 = utils.filter_data_by_labels_with_numbers(x_train, 
                                                          y_train, 
                                                          label_conf)

        y1 = keras.utils.to_categorical(y1, num_total_classes)
        y2 = keras.utils.to_categorical(y2, num_total_classes)
        
        # initialize models
        model1 = get_model_func()
        model2 = keras.models.clone_model(model1)
        model1.set_weights(pretrained_weights)
        model2.set_weights(pretrained_weights)
        compile_model(model1)
        compile_model(model2)

        # fit
        fit_model_with_datasets(model1, local_epochs, x1, y1)
        fit_model_with_datasets(model2, local_epochs, x2, y2)
        
        additionals['l2_dist'] = np.append(additionals['l2_dist'], np.array([semantic_drift.l2_distance(model1, model2)]))

        pre_eval_res = model1.evaluate(x=tx1, y=ty1, verbose=0)
        
        # test
        add_hist_to_dict(res['model_0to4']['test_0to4'],
                         pre_eval_res)
        
        rounds_max = 15

        for ro in range(max_rounds):
            # pick a fraction of local data for training
            p = np.random.permutation(len(x1))
            x1 = x1[p][:fr_data_size]
            y1 = y1[p][:fr_data_size]
            p = np.random.permutation(len(x2))
            x2 = x2[p][:fr_data_size]
            y2 = y2[p][:fr_data_size]

            #aggregate
            weights = [model1.get_weights(), model2.get_weights()]
            agg_weights = list()
            theta = 0.5
            for weights_list_tuple in zip(*weights):
                agg_weights.append(np.array([np.average(np.array(w), axis=0, weights=[1. - theta, theta]) for w in zip(*weights_list_tuple)]))
            aggr_model = keras.models.clone_model(model1)
            aggr_model.set_weights(agg_weights)
            compile_model(aggr_model)
            
            model1.set_weights(agg_weights)
            model2.set_weights(agg_weights)
            
            eval_res = aggr_model.evaluate(x=tx1, y=ty1, verbose=0)
            additionals['loss_benefit'][ro].append((pre_eval_res[0] - eval_res[0]) / pre_eval_res[0])
            print("o", end='')
            fit_model_with_datasets(model1, epochs, x1, y1)
            fit_model_with_datasets(model2, epochs, x2, y2)

        print("")
        
        add_hist_to_dict(res['model_aggr']['test_0to4'],
                         eval_res)
        K.clear_session()
        
    return res, additionals

def multiple_experiments(func, num, params):
    shape = (len(np.arange(1, 0-params['diff'], -params['diff'])), 2)
    res_sum = {}
    start_time = datetime.datetime.now()
    additionals_sum = {}
    for n in range(num):
        print("------------- experiment {} -------------".format(n+1))
        res, additionals = func(**params)
        
        for k in additionals:
            if k not in additionals_sum:
                additionals_sum[k] = [copy.deepcopy(additionals[k])]
            else:
                additionals_sum[k].append(additionals[k])
        
        for k in res: # for(models)
            if k not in res_sum:
                res_sum[k] = copy.deepcopy(res[k])
            else:
                for l in res_sum[k]: # for(test sets)
                    for i in res_sum[k][l]: # for(metric)
                        res_sum[k][l][i] += res[k][l][i]
        
        elasped = datetime.datetime.now() - start_time
        rem = elasped / (n+1) * (num-n-1)
        print("elasped time: {}".format(elasped))
        print("remaining time: {}".format(rem))
        K.clear_session()

    for k in res_sum:
        for l in res_sum[k]:
            for i in res_sum[k][l]:
                res_sum[k][l][i] /= num
          
    return res_sum, additionals_sum

def multiple_experiments_with_diff_variables(func, var_name, iter_list, params):
    res_all = {}
    start_time = datetime.datetime.now()
    additionals_all = []
    i = 0

    for var in iter_list:
        print("------------- experiment {} -------------".format(i+1))
        params[var_name] = var
        res, additionals = func(**params)
        
        additionals_all.append(additionals)
        
        # for k in res: # for(models)
        #     if k not in res_all:
        #         res_all[k] = copy.deepcopy(res[k])
        #     else:
        #         for l in res_all[k]: # for(test sets)
        #             for i in res_all[k][l]: # for(metric)
        #                 res_all[k][l][i] = res[k][l][i]
        
        elasped = datetime.datetime.now() - start_time
        rem = elasped / (i+1) * (len(iter_list) - i -1)
        print("elasped time: {}".format(elasped))
        print("remaining time: {}".format(rem))
        K.clear_session()
        i += 1
          
    return res_all, additionals_all
