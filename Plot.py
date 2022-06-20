from Process_data import load_data
from Model_opt_tools import train_test_layer, extract_final_params
from Opt_layer_by_layer import get_model, set_layer_params
import matplotlib.pyplot as plt
import numpy as np
import json

data_path = "/Users/mariiazamyrova/Documents/htr_optimization/HTR_results/"

metric_path = "/Users/mariiazamyrova/Documents/htr_optimization/HTR_metrics/"

N_layers = 3

# Load data

train_data, train_list, test_data, test_list, val_split = load_data(0.2)

# Get optimal parameters

best_params = [extract_final_params(data_path+'Tuned_layer'+str(n)+'/best_final-layer'+str(n)+".json")["best_params"] for n in range(N_layers)]

model_layers = get_model()

# Uniform parameters

same_params = {'sr': 0.9, 'lr': 0.5, 'input_scaling': 0.5, 'ridge': 0.001}

# Random parameters

random_params = [{"sr": 0.11241733297182535, "lr": 0.01374294755965344, "input_scaling": 0.33176565747743836, "ridge": 0.00015606131642820778},
                 {"sr": 0.11696782141164089, "lr": 0.7245456022318985, "input_scaling": 6.031081625184072, "ridge": 0.013304215588851375},
                 {"sr": 0.3444565825563881, "lr": 0.8834450028421846, "input_scaling": 0.11632960722145523, "ridge": 1.4523764570236597e-08}]

# Original HTR parameters

htr_params = {'sr':[0.8616, 1.1786, 1.2298],
                 'lr':[0.2961, 0.4663, 0.6632], 'input_scaling': [0.7210, 8.8463, 2.6722],
                 'ridge': [6.56*(1e-3), 4.59*(1e-5), 1.49*(1e-5)]}

# Parameter value ranges

test_ranges = {"sr":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

"input_scaling":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

"lr":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

trials_layers = []
best_layers = []

# This function returns predictions for every layer with given parameters

def pred_for_every_layer(tr_d, ts_d, tr_l, ts_l, param_config="opt"):
    all_preds = []
    all_errors = []
    dataset = ((tr_d, tr_l[0]), (ts_d, ts_l[0]))
       
    for n, (tr_tar, ts_tar) in enumerate(list(zip(tr_l[:N_layers], ts_l[:N_layers]))):
        if param_config == "opt":
            set_layer_params(model_layers, n, best_params[n])
        elif param_config == "rand":
            set_layer_params(model_layers, n, random_params[n])
        elif param_config == "uni":
            set_layer_params(model_layers, n, same_params)
        elif param_config == "orig":
            par = {k:htr_params[k][n] for k in htr_params}   
            set_layer_params(model_layers, n, par)
         
        preds, errors = train_test_layer(dataset, model_layers[n])
        dataset = ((preds[0], tr_l[n+1]), (preds[1], ts_l[n+1]))
        all_preds.append(preds)
        all_errors.append(errors)
        
        
    return all_preds, all_errors

   
def plot_time_orig_vs_pred():
    
    preds_layers, errors_layer = pred_for_every_layer(train_data, test_data, train_list, test_list)
         
#plot timescale ratios original
    labels_orig = ["WD/PH original data", "POS/WD original data"]
    labels_pred = ["WD/PH predicted data", "POS/WD predicted data"]
    fig, axs = plt.subplots(2, 2)
    for n, (orig, pr) in enumerate(list(zip(labels_orig, labels_pred))):
        ratios = []
        for ph, wd in zip(test_list[n], test_list[n+1]):
            ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph, axis=0)))[0]))
            wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd, axis=0)))[0]))
            ratios.append(wd_duration / ph_duration)
        
        axs[n][0].hist(ratios, 100)
        axs[n][0].set_title(orig)
        axs[n][0].set_xlabel("ratio")
        #plt.show()
        
        #plot timescale ratios predicted
        
        ratios_pre = []
        for ph, wd in zip(preds_layers[n][1], preds_layers[n+1][1]):
            ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph.T, axis=0)))[0]))
            wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd.T, axis=0)))[0]))
            ratios_pre.append(wd_duration / ph_duration)
        
        
        axs[n][1].hist(ratios_pre, 100)
        axs[n][1].set_title(pr)
        axs[n][1].set_xlabel("ratio")
        #plt.show()
    plt.rcParams['figure.dpi'] = 1200
    fig.tight_layout()
    plt.show()
    
# This function gets the error value from file

def get_errors(par, layer):
    errors = None
    with open(metric_path+"errors_"+par+"_layer_"+str(layer)+"new_final.json") as dic:
        errors = json.load(dic)
    errs = []
    for i in range(N_layers):
        errs_l = [errors[k]["layer"+str(i)]["errors"]["testing error"]*100 for k in errors]
        errs.append(errs_l)

    return errs
    
axis_labels = {"sr": r"$\rho$", "lr": r"$\alpha$", "input_scaling": r"$\sigma$"}

axis_ticks = {"sr": np.arange(0, 11, 1), "lr": np.arange(0, 1.1, 0.1),
              "input_scaling": np.arange(0, 11, 1)  }

# This function plots error rate against parameter value

def plot_param_vs_error(par, layer):
    fig = plt.figure(dpi=1200)
    thick = 4
    style = ['-', '--', '-.']
    for l in range(N_layers):
        errs = get_errors(par, l)
        plt.plot(test_ranges[par], errs[layer], label=axis_labels[par]+str(l+1), linewidth=thick, linestyle=style[l])
        thick -= 1
        
        plt.ylabel("Layer "+str(layer+1)+" FER")
        plt.xlabel(axis_labels[par])
        plt.xticks(axis_ticks[par])
        plt.yticks(np.arange(max(0, np.round(np.min(np.array(errs[layer])), 0)-1), np.round(np.max(np.array(errs[layer])), 0)+1.5, 0.5))
        plt.legend()

# This function plots the temporal ratio distributions

def plot_temp_ratios(useOrigData = True, param_config="opt"):
    
    if useOrigData:
        fig, axs = plt.subplots(1, 2)
        ratios = []
        
        for ph, wd in zip(test_list[0], test_list[1]):
            ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph, axis=0)))[0]))
            wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd, axis=0)))[0]))
            ratios.append(wd_duration / ph_duration)
            
        axs[0].hist(ratios, 100)
        
        ratios = []
        
        for ph, wd in zip(test_list[1], test_list[2]):
            ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph, axis=0)))[0]))
            wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd, axis=0)))[0]))
            ratios.append(wd_duration / ph_duration)
            
        axs[1].hist(ratios, 100)
    
        plt.rcParams['figure.dpi'] = 1200
        fig.tight_layout()
        plt.show()
        
    else:
    
        preds_layers, errors_layer = pred_for_every_layer(train_data, test_data, train_list, test_list, param_config)
        
        fig, axs = plt.subplots(1, 2)
        ratios = []
        
        for ph, wd in zip(preds_layers[0][1], preds_layers[1][1]):
            ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph, axis=0)))[0]))
            wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd, axis=0)))[0]))
            ratios.append(wd_duration / ph_duration)
            
        axs[0].hist(ratios, 100)
        
        ratios = []
        
        for ph, wd in zip(preds_layers[1][1], preds_layers[2][1]):
            ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph, axis=0)))[0]))
            wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd, axis=0)))[0]))
            ratios.append(wd_duration / ph_duration)
            
        axs[1].hist(ratios, 100)
    
        plt.rcParams['figure.dpi'] = 1200
        fig.tight_layout()
        plt.show()
    
# This function plots the temporal ratio distribution boxplots
    
def plot_time_ratios_box(param_config="opt"):
    
    all_ratios_1 = [] # ratios between layers 2 and 1
    all_ratios_2 = [] # ratios between layers 3 and 2
    
    # get ratios from original dataset
    
    ratios = []
    for ph, wd in zip(test_list[0], test_list[1]):
        ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph, axis=0)))[0]))
        wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd, axis=0)))[0]))
        ratios.append(wd_duration / ph_duration) 
    ratios = np.array(ratios)
    all_ratios_1.append(ratios[~np.isnan(ratios)])
    
    ratios = []
    for ph, wd in zip(test_list[1], test_list[2]):
        ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph, axis=0)))[0]))
        wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd, axis=0)))[0]))
        ratios.append(wd_duration / ph_duration)
    ratios = np.array(ratios)
    all_ratios_2.append(ratios[~np.isnan(ratios)])
    
    # get ratios for every parameter configuration
    
    for par in ["opt", "orig", "rand", "uni"]:
        preds_layers, errors_layer = pred_for_every_layer(train_data, test_data, train_list, test_list, par)
        
        ratios = []
        for ph, wd in zip(preds_layers[0][1], preds_layers[1][1]):
            ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph.T, axis=0)))[0]))
            wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd.T, axis=0)))[0]))
            ratios.append(wd_duration / ph_duration)   
        ratios = np.array(ratios)
        all_ratios_1.append(ratios[~np.isnan(ratios)])
        
        ratios = []
        for ph, wd in zip(preds_layers[1][1], preds_layers[2][1]):
            ph_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(ph.T, axis=0)))[0]))
            wd_duration = np.mean(np.diff(np.nonzero(np.diff(np.argmax(wd.T, axis=0)))[0]))
            ratios.append(wd_duration / ph_duration)
        ratios = np.array(ratios)
        all_ratios_2.append(ratios[~np.isnan(ratios)])
    
    # plot ratios between second and first layers
    # with outliers
    
    plt.boxplot(all_ratios_1, flierprops={'marker':'.'})
    plt.rcParams['figure.dpi'] = 1200
    plt.show()
    
    # without outliers
    
    plt.boxplot(all_ratios_1, showfliers=False)
    plt.rcParams['figure.dpi'] = 1200
    plt.show()
    
    # plot ratios between third and second layers
    # with outliers
    
    plt.boxplot(all_ratios_2, flierprops={'marker':'.'})
    plt.rcParams['figure.dpi'] = 1200
    plt.show()
    
    # without outliers
    
    plt.boxplot(all_ratios_2, showfliers=False)
    plt.rcParams['figure.dpi'] = 1200
    plt.sho()



