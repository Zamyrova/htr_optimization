from Process_data import load_data
from Model_opt_tools import objective_single_layer 
from Model_opt_tools import train_test_layer
from Model_opt_tools import extract_params, extract_final_params
import json
from reservoirpy.hyper import research
from reservoirpy.nodes import ESN
import scipy as sp

# Load data

train_data, train_list, test_data, test_list, val_split = load_data(0.2)


N_layers = 3

data_path = "/Users/mariiazamyrova/Documents/htr_optimization/HTR_results/"

data_save_path = "/Users/mariiazamyrova/Documents/htr_optimization/HTR_results/Tuned_layer2/"

metric_path = "/Users/mariiazamyrova/Documents/htr_optimization/HTR_metrics/"

results = {
    "train_data_size":len(train_data),
    "test_data_size":len(test_data)
}

# Generate random parameters

random_params = {'sr':sp.stats.loguniform.rvs(0.1, 10, size=3),
                 'lr':sp.stats.uniform.rvs(size=3), 'iss': sp.stats.loguniform.rvs(0.1, 10, size=3),
                 'ridge': sp.stats.loguniform.rvs(1e-8, 1, size=3)}

# Original HTR parameters

htr_params = {'sr':[0.8616, 1.1786, 1.2298],
                 'lr':[0.2961, 0.4663, 0.6632], 'input_scaling': [0.7210, 8.8463, 2.6722],
                 'ridge': [6.56*(1e-3), 4.59*(1e-5), 1.49*(1e-5)]}

# Initialize model. Has the random parameters by default 

model_layers = [ESN(units=1000, sr=random_params['sr'][0], lr=random_params['lr'][0], 
                    ridge=random_params['ridge'][0], input_scaling=random_params['iss'][0], seed=1234, workers=-1),
                ESN(units=1000, sr=random_params['sr'][1], lr=random_params['lr'][1], 
                                    ridge=random_params['ridge'][1], input_scaling=random_params['iss'][1], seed=1234, workers=-1),
                ESN(units=1000, sr=random_params['sr'][2], lr=random_params['lr'][2], 
                                    ridge=random_params['ridge'][2], input_scaling=random_params['iss'][2], seed=1234, workers=-1)]

# Uniform parameters

same_params = {'sr': 0.9, 'lr': 0.5, 'input_scaling': 0.5, 'ridge': 0.001}

# Ranges for manually tuning parameters

test_ranges = {"sr":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

"input_scaling":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

"lr":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

# This function returns the model

def get_model(): return model_layers

# This function sets the parameters of the specified model layer

def set_layer_params(model, layerN, params):
    for k in params:
        if k == "ridge": 
            model[layerN].nodes[1].set_param(k, params[k])
        else:
            model[layerN].nodes[0].set_param(k, params[k])

# This function runs the full model pipeline

def run_whole_model(tr_d, ts_d, tr_l, ts_l, numRun=0, setPs = False):
    dataset = ((tr_d, tr_l[0]), (ts_d, ts_l[0]))
    
    for n, (tr_tar, ts_tar) in enumerate(list(zip(tr_l[:N_layers], ts_l[:N_layers]))):
        if setPs: 
            params = extract_final_params(data_path+'Tuned_layer'+str(n)+'/best_final-layer'+str(n)+".json")["best_params"]
            set_layer_params(model_layers, n, params)
        preds, errors = train_test_layer(dataset, model_layers[n])
        
        param = {"sr": model_layers[n].nodes[0].get_param('sr'), "lr": model_layers[n].nodes[0].get_param('lr'),
                 "iss": model_layers[n].nodes[0].get_param('input_scaling'), "ridge": model_layers[n].nodes[1].get_param('ridge')}
        
        results["layer"+str(n)] = {"errors": errors, "params": param}
        dataset = ((preds[0], tr_l[n+1]), (preds[1], ts_l[n+1]))
        
    with open(metric_path+"test_train_results_run"+str(numRun)+".json", "+w") as f:
        json.dump(results, f)


# Preparation steps for manually varying parameter values
# Precompute the output of the first and second layers for efficiency

# get layer0 predictions

dataset0 = ((train_data, train_list[0]), (test_data, test_list[0]))
params0 = extract_final_params(data_path+'Tuned_layer'+str(0)+'/best_final-layer'+str(0)+".json")["best_params"]
set_layer_params(model_layers, 0, params0)
preds0, errors0 = train_test_layer(dataset0, model_layers[0])

# get layer1 predictions

dataset1 = ((preds0[0], train_list[1]), (preds0[1], test_list[1]))
params1 = extract_final_params(data_path+'Tuned_layer'+str(1)+'/best_final-layer'+str(1)+".json")["best_params"]
set_layer_params(model_layers, 1, params1)
preds1, errors1 = train_test_layer(dataset1, model_layers[1])

# get data for layer2
dataset2 = ((preds1[0], train_list[2]), (preds1[1], test_list[2]))
params2 = extract_final_params(data_path+'Tuned_layer'+str(2)+'/best_final-layer'+str(2)+".json")["best_params"]

dataset = [dataset0, dataset1, dataset2]
best_params = [params0, params1, params2]
errors_list = [errors0, errors1]

# This function measures layer error rate with various parameter values

def collect_errors(lyr, par):
    
    errs = {}
    for p in test_ranges[par]:
        dset = dataset[lyr]
        sub_err = {}
        if lyr > 0:
            for i in range(lyr):
                sub_err["layer"+str(i)] =  {"errors": errors_list[i], "params": best_params[i]}
                
        for n in range(lyr, N_layers, 1):
            params = best_params[n].copy()
            if n == lyr: 
                params[par] = p
            set_layer_params(model_layers, n, params)
            preds, errors = train_test_layer(dset, model_layers[n])
            
            param = {"sr": model_layers[n].nodes[0].get_param('sr'), "lr": model_layers[n].nodes[0].get_param('lr'),
                     "iss": model_layers[n].nodes[0].get_param('input_scaling'), "ridge": model_layers[n].nodes[1].get_param('ridge')}
            
            sub_err["layer"+str(n)] =  {"errors": errors, "params": param}
            dset = ((preds[0], train_list[n+1]), (preds[1], test_list[n+1]))
            
        errs[str(p)] = sub_err
        
        with open(metric_path+"errors_"+par+"_layer_"+str(lyr)+"new_final.json", "+w") as f:
            json.dump(errs, f)

# This function performs a single iteration of the random search

def random_search_layer(hyperopt_config, dataset, file_mode='w+', layer=0, runN=1):

    with open(data_save_path+hyperopt_config['exp']+"-layer"+str(layer)+"-run"+str(runN)+".config.json", file_mode) as f:
        json.dump(hyperopt_config, f)
    
    dataset_tune = ((dataset[0][0][:val_split], dataset[0][1][:val_split]), (dataset[0][0][val_split:], dataset[0][1][val_split:]))

    best, trials = research(objective_single_layer, dataset_tune, data_save_path+hyperopt_config['exp']+"-layer"+str(layer)+"-run"+str(runN)+".config.json", 
                            data_save_path+hyperopt_config['exp']+"-layer"+str(layer)+"-run"+str(runN))

    print("\nLayer"+str(layer)+" has been tuned\n")
    
    best = extract_params(data_save_path+'hyperopt-multiscroll-layer'+str(layer)+'-run'+str(runN)+'/hyperopt-multiscroll/results')

    with open(data_save_path+"best-layer"+str(layer)+"-run"+str(runN)+".json", file_mode) as f:
        json.dump(best, f)
 
# This function performs the manual tuning of the ridge regularization

def tune_ridge(layer, dataset):
    
    dataset_tune = ((dataset[0][0][:val_split], dataset[0][1][:val_split]), (dataset[0][0][val_split:], dataset[0][1][val_split:]))

    ridge_range = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1] 
    
    min_ind = 0
    min_error = 20
          
    for n, r in enumerate(ridge_range):
        set_layer_params(model_layers, layer, {"ridge":r})
        preds, errors = train_test_layer(dataset_tune, model_layers[layer])
        if errors["testing error"] < min_error:
            min_ind = n
            min_error = errors["testing error"]
            
    return ridge_range[min_ind]
 
# This function performs full optimization of a single layer

def perform_search(layerN):
    
    dataset = ((train_data, train_list[0]), (test_data, test_list[0]))
    
    # if the layer is not first, get input data from previous layers
    if layerN > 0:
        for n, (tr_tar, ts_tar) in enumerate(list(zip(train_list[:layerN], test_list[:layerN]))):
            params = extract_final_params(data_path+'Tuned_layer'+str(n)+'/best_final-layer'+str(n)+".json")["best_params"]
            set_layer_params(model_layers, n, params)
            preds, errors = train_test_layer(dataset, model_layers[n])
            dataset = ((preds[0], train_list[n+1]), (preds[1], test_list[n+1]))
    
    # tune the spectral radius and the leaky integrator
    print("\nTuning sr and lr:\n")
    hyperopt_config1 = {
        "exp": f"hyperopt-multiscroll", 
        "hp_max_evals": 100,             
        "hp_method": "random",           
        "seed": 42,                     
        "instances_per_trial": 5,        
        "hp_space": {                    
            "N": ["choice", 100],             
            "sr": ["loguniform", 0.1, 10],   
            "lr": ["uniform", 0.1, 1],  
            "iss": ["choice", 1],           
            "ridge": ["loguniform", 1e-8, 1],        
            "seed": ["choice", 1234]          
            }
    }
       
    random_search_layer(hyperopt_config=hyperopt_config1, dataset=dataset, layer=layerN, runN=1)    
        
    best = extract_final_params(data_save_path+"best-layer"+str(layerN)+"-run"+str(1)+".json")['current_params']
    print("Best spectral radius: "+str(best['sr']))
    print("Best leaky integrator: "+str(best['lr']))
    
    # tune the input norm
    
    print("\nTuning iss:\n")
    hyperopt_config2 = {
        "exp": f"hyperopt-multiscroll", 
        "hp_max_evals": 100,             
        "hp_method": "random",           
        "seed": 42,                      
        "instances_per_trial": 5,        
        "hp_space": {                  
            "N": ["choice", 100],             
            "sr": ["choice", best['sr']],   
            "lr": ["choice", best['lr']],  
            "iss": ["loguniform", 0.1, 10],           
            "ridge": ["loguniform", 1e-8, 1],        
            "seed": ["choice", 1234]         
            }
    }
       
    random_search_layer(hyperopt_config=hyperopt_config2, dataset=dataset, layer=layerN, runN=2)    
    
    best_iss = extract_final_params(data_save_path+"best-layer"+str(layerN)+"-run"+str(2)+".json")['current_params']['iss']
    
    params = {'sr':best['sr'], 'lr':best['lr'], 'input_scaling':best_iss}
    
    set_layer_params(model_layers, layerN, params)
    
    # tune the ridge regularization
    
    print("\nTuning ridge:\n")
    best_ridge = tune_ridge(layerN, dataset)
    
    params['ridge'] = best_ridge
    
    set_layer_params(model_layers, layerN, params)
    
    # measure performance with optimized parameters
    
    preds, errors = train_test_layer(dataset, model_layers[layerN])
    
    results["training_error"] = errors["training error"]
    
    results["testing_error"] = errors["testing error"]
    
    results["best_params"] = params
    
    with open(data_save_path+"best_final-layer"+str(layerN)+".json", "+w") as f:
        json.dump(results, f)
    

        
    