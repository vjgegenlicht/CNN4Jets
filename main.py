import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse

from util import DataSet, load_data
from cnn import cnn


def main(params):
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--classify', type=str)
    args    = parser.parse_args()
    
    data_store  = {}
    
    use_cuda    = torch.cuda.is_available()
    device      = torch.device("cuda:0" if use_cuda else "cpu")
    data_store["device"] = device
    
    # Load the data
    print('» Loading data.')
    data    = {}
    classes = []
    for set in list(params.get('input_paths',{}).keys()):
        data[set] = load_data(params['input_paths'][set])
        classes.append(set)
    n_classes   = len(classes) 
    
    # Create the CNN model
    print('» Creating model.')
    model = cnn(n_classes, params, data_store)

    # Create output folder
    output_path = params['output_path']
    os.makedirs(output_path, exist_ok=True)
    
    if args.classify is None:
        # Load the model
        if not args.model is None:
            print('» Loading model state.')
            model.load(args.model)

        # Make data set
        print('» Creating dataset.')
        Dataset     = DataSet(data, classes, params, data_store)    
        
        #Set optimizer
        print('» Setting optimizer.')
        model.set_optimizer(Dataset)
    
        # Train the model
        print('» Training model.')
        model.train(Dataset)

        # Generate predictions for the test data set
        #est_true_y, test_predict_prob, test_predict_y = model.predict_test(Dataset)
        
        # Save the model
        model.save()
        
    else:
        if args.model is None:
            raise ImportError('Please specify a model for the classification using the argument [--model=<path>].')
        
        # Load model
        print('» Loading model state.')
        model.load(args.model, load_optim=False)
        
        # Load the instance
        print('» Loading Instance')
        instance = np.load(args.classify)
        try:
            instance = torch.from_numpy(instance)
        except:
            pass
        
        # Reshape correctly
        n_pixel     = params['n_pixel']
        instance    = instance.reshape(1,1,n_pixel,n_pixel)
        
        # Compute probabilities
        print('» Estimate probabilities')
        prob = model.predict_instance(instance).detach().cpu().numpy().squeeze()
        print(f'⤷ Probability:\n \
        {prob}')
        np.save(output_path + '/tmp_prob', prob)
    
    print('» Label to class translation: \n \
        {}'.format(data_store['label2set']))
    
params = {
'input_paths'           : {'ttbar'    : './conv_out_ttbar.txt',
                           'qcdJets'  : './conv_out_qcdJets.txt'},
'output_path'           : './model',
'image_output_path'     : './images',

'intensity_measure'     : 'pt',
'n_pixel'               : 40,
'eta_range'             : [-1, 1],
'phi_range'             : [-np.pi, np.pi],

'jet_preproc_steps'     : ['phi_alignment','centering'],
'image_preproc_steps'   : ['normalization'],
'train_split'           : 0.995,
'batch_size'            : 1000, 

'n_images'              : 10,

'lr'                    : 1.e-3,
'betas'                 : [0.9,0.99],
'L2'                    : 1.e-3,
'p_drop'                : 0,

'n_epochs'              : 10
}

main(params)