from reservoirpy.nodes import ESN
import numpy as np
from hyperopt import STATUS_OK 
from os import listdir
from os.path import isfile, join
import json

# This functions computes the Frame Error Rate

def FER(y_pred, y_test):
    total = 0
    num_incorrect = 0
    for pr, ts in list(zip(y_pred, y_test)):
        num_incorrect += np.sum(np.argmax(pr, axis=0) != np.argmax(ts, axis=0))
        total += pr.shape[1]
  
    return num_incorrect/total
    
# This is the objective function for optimizing a single layer

def objective_single_layer(dataset, config, *, iss, N, sr, lr, ridge, seed):

    train_data, test_data = dataset
    X_train, y_train = train_data
    X_test, y_test = test_data

    instances = config["instances_per_trial"]

    variable_seed = seed

    losses = []
  
    for n in range(instances):

        layer = ESN(units=N, sr=sr, lr=lr, ridge=ridge,
                           input_scaling=iss, seed=variable_seed, workers=-1)

        target = [t.T for t in y_train]
        layer.fit(X_train, target)

        test_pred = [layer.run(ts) for ts in X_test]
        
        variable_seed += 1
      
        predictions = [t.T for t in test_pred]
      
        loss = FER(predictions, y_test)

        losses.append(loss)

    return {'loss': np.mean(losses),
           'status': STATUS_OK}

# This function computes the train and test data predictions and errors

def train_test_layer(dataset, layer):
    
    train_data, test_data = dataset
    X_train, y_train = train_data
    X_test, y_test = test_data

    #train and test
    target_ts = [t.T for t in y_train]
    layer.fit(X_train, target_ts)
    
    train_pred = [layer.run(tr) for tr in X_train]
    train_error = FER([t.T for t in train_pred], y_train)

    test_pred = [layer.run(ts) for ts in X_test]
    test_error = FER([t.T for t in test_pred], y_test)
    
    preds = [train_pred, test_pred]
    errors = {
        "training error" : train_error,
        "testing error" : test_error
        }
    
    return preds, errors

# This function computes the training, validation and test errors

def train_val_test_layer(dataset, val_split, params):
    
    train_data, test_data = dataset
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    layer = ESN(units=params['N'], sr=params['sr'], lr=params['lr'], ridge=params['ridge'],
                       input_scaling=params['iss'], seed=params['seed'], workers=-1)

    #train and validate
    target = [t.T for t in y_train[:val_split]]
    layer.fit(X_train[:val_split], target)
    train_pred = [layer.run(tr) for tr in X_train[:val_split]]

    train_error = FER([t.T for t in train_pred], y_train[:val_split])

    val_pred = [layer.run(tr) for tr in X_train[val_split:]]

    val_error = FER([t.T for t in val_pred], y_train[val_split:])

    #train and test
    target_ts = [t.T for t in y_train]
    layer.fit(X_train, target_ts)
    
    train_pred_whole = [layer.run(tr) for tr in X_train]

    test_pred = [layer.run(ts) for ts in X_test]
    test_error = FER([t.T for t in test_pred], y_test)
    
    preds = [train_pred_whole, test_pred, train_pred, val_pred]
    errors = {
        "training error" : train_error,
        "validation error" : val_error,
        "testing error" : test_error
        }
    
    return preds, errors
    
# This function extracts the best parameters given a file with all of the 
# search trials

def extract_params(directory):
    trials = []
    for f in listdir(directory): 
        file = join(directory, f)
        if isfile(file):
            with open(file) as dic:
                trials.append(json.load(dic))

    trials.sort(key = (lambda x: x['returned_dict']['loss']))
    best = trials[0]
    return best

# This function extracts data for every trial from file

def extract_all_trials(directory):
    trials = []
    for f in listdir(directory): 
        file = join(directory, f)
        if isfile(file):
            with open(file) as dic:
                trials.append(json.load(dic))

    return trials

# This function extracts parameters from file

def extract_final_params(file):
    params = None
    with open(file) as dic:
        params = json.load(dic)

    return params

        
        
        
        
        
        
        
        
        
