import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


### SET PATHS
input_base_path = "/scratch/schatter/ExplainCOVID/Activations/AllTestFold0/0 Specific Subjects/"
output_base_path = "/project/ankrug/ExplainCOVID/test_complete/" 


def load_example_info(file):
    example_info = np.load(file,
                      allow_pickle=True)[()]
    inp = example_info['input']
    tgt = example_info['target']
    act = example_info['activations']
    return inp, tgt, act

def get_model_files_and_layers(model):
    model_suffix = '_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray/'
    
    files = os.listdir(os.path.join(input_base_path, model + model_suffix))
    model_files = [base_path + model + model_suffix + file for file in files]
    
    _, _, act = load_example_info(model_files[0])
    layers_of_interest = [*act.keys()]
    
    return model_files, layers_of_interest

def get_absmax_clim(ndarray):
    clim = np.max(np.abs(ndarray))
    clim = (-clim,clim)
    return clim

models = ['DenseNet161', 'IncepResV2', 'InceptionV3', 'ResNet18', 'ResNet34']
groups_names = ["COVID-19\n", "Pneumonia\n", "Pneumonia\nwithout COVID-19", "Healthy\n"]


### Find dead neurons in each layer

print("Identifying dead neurons...")

dead_neurons_path = os.path.join(output_base_path, "dead_neurons")
if not os.path.isdir(dead_neurons_path):
    os.makedirs(dead_neurons_path)

for model in models:
    files, layers_of_interest = get_model_files_and_layers(model)

    zero_range_ids = dict()
    for layername in layers_of_interest:
        zero_range_ids[layername] = list()

    for file in tqdm(files, desc=model):
        inp, tgt, act = load_example_info(file)

        for layername in layers_of_interest:
            fms = act[layername][0]

            min_values = np.min(fms,axis=(1,2))
            max_values = np.max(fms,axis=(1,2))
            value_range = max_values - min_values

            is_zero_range = value_range <= 1e-30
            if len(zero_range_ids[layername])==0:
                zero_range_ids[layername] = np.argwhere(is_zero_range)[:,0]
            else:
                zero_range_ids[layername] = np.intersect1d(zero_range_ids[layername], np.argwhere(is_zero_range)[:,0])

    np.save(os.path.join(dead_neurons_path, model + ".npy"), zero_range_ids)
                
    for layername in layers_of_interest:
        fms = act[layername][0]
        n_fms = len(fms)
        print(str(np.round(100*len(zero_range_ids[layername])/n_fms, 2)) + " % of feature maps inactive for all images in", layername)
        
        
### stacking activations of all examples

print("Stacking example activations...")

stacked_activations_path = os.path.join(output_base_path, "stacked_activations")
if not os.path.isdir(stacked_activations_path):
    os.makedirs(stacked_activations_path)

files, layers_of_interest = get_model_files_and_layers('ResNet18')
inp, tgt, _ = load_example_info(files[0])
stacked_targets = np.zeros(shape = [len(files), tgt.shape[1]])
stacked_inputs = np.zeros(shape = [len(files)] + list(inp.shape[1:]))
for file_id, file in enumerate(tqdm(files, desc='tgts,inps')):
    inp, tgt, _ = load_example_info(file)
    stacked_targets[file_id] = tgt[0]
    stacked_inputs[file_id] = inp[0]
np.save(os.path.join(stacked_activations_path, "stacked_targets.npy"), 
        stacked_targets)
np.save(os.path.join(stacked_activations_path, "stacked_inputs.npy"), 
        stacked_inputs)
    
for model in models:
    files, layers_of_interest = get_model_files_and_layers(model)

    dead_neuron_ids = np.load(os.path.join(dead_neurons_path, model + ".npy"), allow_pickle=True)[()]
    _, _, act = load_example_info(files[0])
    active_neuron_ids = dict()
    for layername in layers_of_interest:
        fms = act[layername][0]
        n_fms = len(fms)
        active_neurons = np.arange(n_fms)
        active_neurons = np.setdiff1d(active_neurons, dead_neuron_ids[layername])
        active_neuron_ids[layername] = active_neurons
    
    for layer_id, layername in enumerate(layers_of_interest):
        stacked_layer_activations = np.zeros(shape=[len(files), len(active_neuron_ids[layername])] + list(act[layername].shape)[2:])
        for file_id, file in enumerate(tqdm(files, desc=model)):
            _, _, act = load_example_info(file)
            
            fms = act[layername][0]
            fms = fms[active_neuron_ids[layername]]
            stacked_layer_activations[file_id] = fms
        np.save(os.path.join(stacked_activations_path, model + "_l" + str(layer_id) + ".npy"),
                stacked_layer_activations)
        

### compute average activations over the groups of interest

print("Computing group averages...")

group_average_path = os.path.join(output_base_path, "group_averages")
if not os.path.isdir(group_average_path):
    os.makedirs(group_average_path)
    
inputs = np.load(os.path.join(stacked_activations_path, "stacked_inputs.npy"))
targets = np.load(os.path.join(stacked_activations_path, "stacked_targets.npy"))

# covid (equals covid+pneumonia)  2  1  (10 1)
# pneumonia                       2 any 10 1 
# pneumonia without covid         2  0  10 1
# no finding                      2  0  10 0

groups_ids = list()
groups_ids.append(np.argwhere(targets[:,2]==1)[:,0])
groups_ids.append(np.argwhere(targets[:,10]==1)[:,0])
groups_ids.append(np.argwhere(np.logical_and(targets[:,2]==0, 
                                            targets[:,10]==1))[:,0])
groups_ids.append(np.argwhere(np.logical_and(targets[:,2]==0, 
                                            targets[:,10]==0))[:,0])


input_averages = np.zeros(shape=[len(groups_ids)] + list(inputs.shape)[1:])

for array_idx, group_ids in enumerate(groups_ids):
    #input average
    subset = inputs[group_ids]
    subset_avg = np.mean(subset,0)
    input_averages[array_idx] = subset_avg
        
np.save(os.path.join(group_average_path, "inputs.npy"),
        input_averages)
        
for model in models:
    _ , layers_of_interest = get_model_files_and_layers(model)
    for layer_id, layer in enumerate(layers_of_interest):
        stacked_acts = np.load(os.path.join(stacked_activations_path, model + "_l" + str(layer_id) + ".npy"))
        act_averages = np.zeros(shape=[len(groups_ids)] + list(stacked_acts.shape)[1:])
        
        for array_idx, group_ids in enumerate(groups_ids):
            #act average per layer
            subset = stacked_acts[group_ids]
            subset_avg = np.mean(subset,0)
            act_averages[array_idx] = subset_avg
                
        np.save(os.path.join(group_average_path, model + "_l" + str(layer_id) + ".npy"),
                act_averages)
        
        
### Create figures based on the averaging results

print("Creating figures...")

figure_output_path = os.path.join(output_base_path, "figures")
if not os.path.isdir(figure_output_path):
    os.makedirs(figure_output_path)
    
input_averages = np.load(os.path.join(group_average_path, "inputs.npy"))
input_averages = np.transpose(input_averages, [0,2,3,1])

n_cols = 4
n_rows = 1
fig = plt.figure(figsize=[20,5])
widths = n_cols * [1]
heights = n_rows * [1]
spec = fig.add_gridspec(ncols=n_cols,
                        nrows=n_rows,
                        width_ratios=widths,
                        height_ratios=heights)

for group_id, group_name in enumerate(groups_names):
    ax = fig.add_subplot(spec[0,group_id])
    ax.imshow(input_averages[group_id])
    ax.set_title(group_name)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.savefig(os.path.join(figure_output_path, "input_averages.pdf"), 
            bbox_inches='tight')
plt.show()

n_cols = 4
n_rows = 1
fig = plt.figure(figsize=[20,5])
widths = n_cols * [1]
heights = n_rows * [1]
spec = fig.add_gridspec(ncols=n_cols,
                        nrows=n_rows,
                        width_ratios=widths,
                        height_ratios=heights)

normalizer = np.mean(input_averages, 0)
normalized_averages = input_averages - normalizer
normalized_averages = np.mean(normalized_averages, 3)
clim = get_absmax_clim(normalized_averages)

for group_id, group_name in enumerate(groups_names):
    ax = fig.add_subplot(spec[0,group_id])
    ax.imshow(normalized_averages[group_id], clim=clim, cmap='bwr')
    ax.set_title(group_name)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.savefig(os.path.join(figure_output_path, "input_averages_normalized.pdf"), 
            bbox_inches='tight')
plt.show()

for model in models:
    _ , layers_of_interest = get_model_files_and_layers(model)
    for layer_id, layername in enumerate(layers_of_interest):
        average_acts = np.load(os.path.join(group_average_path, model + "_l" + str(layer_id) + ".npy"))
        normalized_acts = average_acts - np.mean(average_acts,0)
        fm_order = np.argsort(np.mean(np.abs(normalized_acts),(0,2,3)))[::-1]
        
        average_acts = average_acts[:,fm_order]
        normalized_acts = normalized_acts[:,fm_order]
        
        print(model, ":", layername)
        
        n_fms = average_acts.shape[1]
        if n_fms > 10:
            n_fms = 10
        
        n_cols = 10
        n_rows = 4
                
        
        fig = plt.figure(figsize=[18,7])
        widths = n_cols * [1]
        heights = n_rows * [1]
        spec = fig.add_gridspec(ncols=n_cols,
                                nrows=n_rows,
                                width_ratios=widths,
                                height_ratios=heights)
        
        clim = get_absmax_clim(normalized_acts)
        
        for group_id, group_name in enumerate(groups_names):
            for fm_id in range(n_fms):
                ax = fig.add_subplot(spec[group_id,fm_id])
                ax.imshow(normalized_acts[group_id,fm_id], clim=clim, cmap='bwr')
                ax.set_xticks([])
                ax.set_yticks([])
#         plt.tight_layout()
        
        plt.savefig(os.path.join(figure_output_path, "NAPs_" + model + "_l" + str(layer_id) + ".pdf"))
        plt.clos(fig)