import numpy as np
import random
import torch
import torchvision
import pickle

def moving_average(l,window):
    return [sum(l[idx:idx+window])/window for idx in range(len(l)-window+1)]
# seed 
def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

# Methods to save model checkpoints

def save_checkpoint(state, checkpoint_path):
    print("Saving checkpoint ... ")
    torch.save(state, checkpoint_path)
    print("Checkpoint:", checkpoint_path, "saved.")

def load_checkpoint(model, optimizer, load_checkpoint_path):
    print("Loading checkpoint ... ")
    checkpoint = torch.load(load_checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if('optimizer' in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, start_epoch

# Methods to save results

def save_results(results, file):
    performance = results
    with open(file, 'wb') as handle:
        pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def load_results(filename):
    with open(filename, 'rb') as handle:
        performance_history = torch.load(handle)['performance_history']
    return performance_history
