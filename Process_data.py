import numpy as np
import pickle
import scipy.io as spio

data_path = "/Users/mariiazamyrova/Downloads/hierarchical-task-reservoir-srl_cleaned/dataset_srl/"
path = "/Users/mariiazamyrova/Downloads/hierarchical-task-reservoir-srl_cleaned/"
save = True

input_data = spio.loadmat(data_path+"inputs.mat", squeeze_me=True)
inputs = [inp.T for inp in input_data['inputs']]
if save:
    with open(path+"processed_data/inputs.pkl", 'wb') as fp:
        pickle.dump(inputs, fp)
        
# Phone targets
ph_data = spio.loadmat(data_path+"targets_ph.mat", squeeze_me=True)
ph_target_labels = [inp for inp in ph_data['targets_PH']]

# one-hot encoding
Target_indexes = np.array(list(map(int, set(np.concatenate(ph_target_labels)))))-1
I = np.eye(Target_indexes.shape[0])
targets_2 = []
for target in ph_target_labels:
    targets_2.append(np.transpose(I[np.array(list(map(int, target)))-1]))
ph_targets = targets_2

if save:
    with open(path+"processed_data/ph_targets.pkl", 'wb') as fp:
        pickle.dump(ph_targets, fp)
  
# Word targets
wd_data = spio.loadmat(data_path+"targets_wd.mat", squeeze_me=True)
wd_targets = [inp for inp in wd_data['targets_WD']]
if save:
    with open(path+"/processed_data/wd_targets.pkl", 'wb') as fp:
        pickle.dump(wd_targets, fp)
        
# POS targets
pos_dataset = spio.loadmat(data_path+'targets_pos.mat', squeeze_me=True)
pos_targets = [inp for inp in pos_dataset['targets']]
if save:
    with open(path+"/processed_data/pos_targets.pkl", 'wb') as fp:
        pickle.dump(pos_targets, fp)
  
# SRL targets
N_verbs = 6
N_targets = 27
srl_targets = []
for i in range(N_verbs):
    dataset = spio.loadmat(data_path+'targets'+str(i)+'_srl', squeeze_me=True)
    # one-hot encoding
    tar = dataset['targets'+str(i)]
    #Target_indexes = np.array(list(map(int, set(np.concatenate(tar)))))-1
    I = np.eye(N_targets)
    targets_2 = []
    for target in tar:
        targets_2.append(np.transpose(I[np.array(list(map(int, target)))-1]))
    tar = targets_2
    srl_targets.append(tar)
if save:
    with open(path+"/processed_data/srl_targets.pkl", 'wb') as fp:
        pickle.dump(srl_targets, fp)
        
data_path = path+"/processed_data/"
data_labels = ["inputs", "ph_targets", "wd_targets", "pos_targets", "srl_targets"]
dataset = {}
for dl in data_labels:
    with open(data_path+dl+'.pkl', 'rb') as fp:
        data = pickle.load(fp)
    dataset.update({dl: data})
    
data_size = len(dataset['inputs']) 

# This function retrieves the dataset of the specified size
def load_data(cutoff, start = 0):
    
    data_cutoff = int(np.around(data_size * cutoff))

    train = int(0.7 * data_cutoff)
    
    val_split = int(0.7 * train)

    train_data = dataset['inputs'][start:][:train]
      
    train_target1 = dataset["ph_targets"][start:][:train]
       
    train_target2 = dataset["wd_targets"][start:][:train]
       
    train_target3 = dataset["pos_targets"][start:][:train]
      
    train_target4 = [d[start:][:train] for d in dataset["srl_targets"]]
        
    train_list = [train_target1, train_target2, train_target3, train_target4] 
    
    test_data = dataset['inputs'][start:][train:data_cutoff]
    
    test_target1 = dataset["ph_targets"][start:][train:data_cutoff]
    
    test_target2 = dataset["wd_targets"][start:][train:data_cutoff]
    
    test_target3 = dataset["pos_targets"][start:][train:data_cutoff]
    
    test_target4 = [d[start:][train:data_cutoff] for d in dataset["srl_targets"]]
    
    test_list = [test_target1, test_target2, test_target3, test_target4]
    
    return train_data, train_list, test_data, test_list, val_split










