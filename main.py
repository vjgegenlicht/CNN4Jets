import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import image
import os
import argparse
from datetime import datetime
import logging

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
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    params["output_path"] = "./output/" + dt_string + params["output_path"]

    os.makedirs(params["output_path"] + "/log.log", exist_ok=True)
    logging.basicConfig(filename = params["output_path"] + 'log.log', level=logging.INFO)
    logging.info(f"Parameters: \n {params}")
    
    # Load the data
    logging.info('» Loading data.')
    data    = {}
    classes = []
    for set in list(params.get('input_paths',{}).keys()):
        data[set] = load_data(params['input_paths'][set])
        classes.append(set)
    n_classes   = len(classes) 
    
    # Create the CNN model
    logging.info('» Creating model.')
    model = cnn(n_classes, params, data_store)
    model.to(device)

    # Create output folder
    output_path = params['output_path']
    os.makedirs(output_path, exist_ok=True)
    
    if args.classify is None:
        # Load the model
        if not args.model is None:
            logging.info('» Loading model state.')
            model.load(args.model)

        # Make data set
        logging.info('» Creating dataset.')
        Dataset     = DataSet(data, classes, params, data_store)    
        
        #Set optimizer
        logging.info('» Setting optimizer.')
        model.set_optimizer()
        model.set_scheduler(Dataset)
    
        # Train the model
        logging.info('» Training the model.')
        model.train(Dataset)

        # Generate predictions for the test data set
        #est_true_y, test_predict_prob, test_predict_y = model.predict_test(Dataset)
        
        # Save the model
        logging.info('» Saving the model.')
        model.save()
        
    else:
        if args.model is None:
            raise ImportError('Please specify a model for the classification using the argument [--model=<path>].')
        
        # Load model
        logging.info('» Loading model state.')
        model.load(args.model, load_optim=False)
        
        # Load the instance
        logging.info('» Loading Instance')
        try:
            instance = np.load(args.classify)
        except:
            instance = np.array(image.imread(args.classify))

        try:
            instance = torch.from_numpy(instance)
        except:
            pass
        
        # Reshape correctly
        n_pixel     = params['n_pixel']
        instance    = instance.reshape(1,1,n_pixel,n_pixel)
        
        # Compute probabilities
        logging.info('» Estimate probabilities')
        prob = model.predict_instance(instance).detach().cpu().numpy().squeeze()
        logging.info(f'⤷ Probability:\n \
        {prob}')
        np.save(output_path + '/tmp_prob', prob)
    
    logging.info('» Label to class translation: \n \
        {}'.format(data_store['label2set']))
    
params = {
'input_paths'           : {'ttbar'    : 'Data/events_ttbar.txt',
                           'qcdJets'  : 'Data/events_qcdjets.txt',
                           'photons'  : 'Data/events_photons.txt'},
'output_path'           : '/model',
'image_output_path'     : './images',

'intensity_measure'     : 'pt',
'n_pixel'               : 40,
'eta_range'             : [-2, 2],
'phi_range'             : [-np.pi, np.pi],

'jet_preproc_steps'     : ['phi_alignment','centering'],
'image_preproc_steps'   : ['normalization'],
'train_split'           : 0.999,
'batch_size'            : 512, 

'n_images'              : 0,
'output_layers'         : False,
'plot_test'             : True, 

'lr'                    : 8.e-5,
'lr_scheduler'          : 'one_cycle_lr',
'betas'                 : [0.9,0.99],
'L2'                    : 0.0,
'p_drop'                : 0.0,

'n_epochs'              : 1500
}

main(params)