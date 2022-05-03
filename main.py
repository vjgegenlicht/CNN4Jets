import numpy as np
import torch

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import image
import os
import argparse
from datetime import datetime
import logging
import torchvision.transforms as tvtrfs

from util import Rescale, Center
from cnn import cnn


def main(params):
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--classify', type=str)
    args    = parser.parse_args()
    
    data_store  = {}
    
    use_cuda    = torch.cuda.is_available()
    device      = torch.device("cuda:0" if use_cuda else use_cuda)
    data_store["device"] = device
    
    # Create output folder
   # now = datetime.now()
   # dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
    params["output_path"] = "./output/" + params["output_path"]
    os.makedirs(params["output_path"] + "/log.log", exist_ok=True)
    logging.basicConfig(filename = params["output_path"] + 'log.log', level=logging.INFO, filemode='w')
    logging.info(f"Parameters: \n {params}")
    output_path = params['output_path']
    os.makedirs(output_path, exist_ok=True)

    # Create the CNN model
    logging.info('» Creating model.')
    model = cnn(params, data_store)
    model.to(device)
    
    
    if args.classify is None:
        if not args.model is None:
            # Load the model
            logging.info('» Loading model state.')
            model.load(args.model)
        else:
            # Initialize the model
            logging.info('» Initializing the model.')
            model.initialize_dataloaders()
            model.set_optimizer()
            model.set_scheduler()
    
        # Train the model
        logging.info('» Training the model.')
        model.train()
        model.train_model()
        
        # Save the model
        logging.info('» Saving the model.')
        model.save()
    
    elif args.classify == "test":
        # Load the model
        logging.info('» Loading model state.')
        model.load(args.model, load_optim=False, load_dataloaders=True)

        model.eval()
        model.plot_test()

    else:
        if args.model is None:
            raise ImportError('Please specify a model for the classification using the argument [--model=<path>].')

        # Load the instance
        logging.info('» Loading Instance')
        
        # Load the model
        logging.info('» Loading model state.')
        model.load(args.model, load_optim=False, load_dataloaders=False)

        logging.info("⤷ Loading image as PNG")
        instance    = Image.open(args.classify) #[:,:,0]
        transforms  = tvtrfs.Compose([tvtrfs.Grayscale(num_output_channels=1),
                                        Center(n_pixel=params["n_pixel"]),
                                        tvtrfs.ToTensor(),
                                        Rescale(norm=1)])
        instance    = transforms(instance)
        instance = instance.to(device=device)
                
        # Reshape correctly
        n_pixel     = params['n_pixel']
        instance    = instance.reshape(1,1,n_pixel,n_pixel)

        # Set model in evaluation mode
        model.eval()        
        
        # Compute probabilities
        logging.info('» Estimate probabilities')
        prob = model.predict_instance(instance).detach().cpu().numpy().squeeze()
        logging.info(f'⤷ Probability:\n {prob}')
        # np.save(output_path + '/tmp_prob', prob)
        np.savetxt(output_path + '/probabilites.txt', prob)
    


    logging.info('» Label to class translation: \n \
        {}'.format(data_store['label2set']))

    
    
params = {
'input_paths'           : {'ttbar'    : 'Data/events_ttbar.txt',
                           'qcdJets'  : 'Data/events_qcdjets.txt',
                           'photons'  : 'Data/events_photons.txt',
                           'smileys'  : 'Data/smileys.npy'},
'n_classes'             : 4,
'output_path'           : '/model',
'image_output_path'     : './images',

'intensity_measure'     : 'pt',
'n_pixel'               : 40,
'eta_range'             : [-2, 2],
'phi_range'             : [-np.pi, np.pi],

'jet_preproc_steps'     : ['phi_alignment','centering'],
'image_preproc_steps'   : ['normalization'],
'train_split'           : 0.95,
'batch_size'            : 512, 

'n_images'              : 0,
'output_layers'         : False,
'plot_test'             : True, 
'check_calibration'     : False,

'lr'                    : 5.e-4,
'lr_scheduler'          : 'one_cycle_lr',
'betas'                 : [0.9,0.99],
'L2'                    : 0.0,
'p_drop'                : 0.3,

'n_epochs'              : 100
}

main(params)