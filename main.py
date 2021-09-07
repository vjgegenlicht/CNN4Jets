import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from util import DataSet, load_data
from cnn import cnn


def main(params):
    
    data_store  = {}
    
    use_cuda    = torch.cuda.is_available()
    device      = torch.device("cuda:0" if use_cuda else "cpu")
    data_store["device"] = device

    # Load the data
    data    = {}
    classes = []
    for set in list(params.get('input_paths',{}).keys()):
        data[set] = load_data(params['input_paths'][set])
        classes.append(set)

    Dataset     = DataSet(data, classes, params, data_store)
    
    n_classes   = len(classes) 
    
    model = cnn(n_classes, params, data_store)
    
    model.train(Dataset)


    

params = {
'input_paths'           : {'ttbar'    : './conv_out_ttbar.txt',
                           'qcdJets'  : './conv_out_qcdJets.txt'},
'output_path'           : './images_out.dat',
'image_output_path'     : './images',

'intensity_measure'     : 'pt',
'n_pixel'               : 40,
'eta_range'             : [-4, 4],
'phi_range'             : [-np.pi, np.pi],

'jet_preproc_steps'     : ['phi_alignment','centering'],
'image_preproc_steps'   : ['normalization'],
'train_split'           : 0.7,
'batch_size'            : 1000, 

'n_images'              : 10,

'lr'                    : 1.e-3,
'betas'                 : [0.9,0.99],
'L2'                    : 1.e-3,
'p_drop'                : 0,

'n_epochs'              : 100
}

main(params)