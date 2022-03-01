#from itertools import Predicate
import numpy as np
import torch
from constit2images import *
import logging

def load_data(input_path, set):
    '''Load the data from the given path and transforms it to a 3d numpy array [n_jets, n_constituents, 4].
        args: 
            input_path:     : [string]  path to the txt file holding the data
    '''
    if not set == "smileys":
        jets    = np.loadtxt(input_path)
        n_jets  = jets.shape[0]
        n_const = int(jets.shape[1]/4)
        jets    = jets.reshape(n_jets, n_const, 4)
    else:
        jets    = np.load(input_path, allow_pickle=True)
    return jets

class DataSet:
    def __init__(self, data, classes, params, data_store):
        '''Initializes a data set from the detector images.
            args:
                data    : [dict]    dictionary holding (class, data) pairs
                classes : [list]    list holding the strings of classes
        '''
        # Parameters
        self.n_classes  = len(classes)
        self.params     = params
        self.device     = data_store['device']
        self.classes    = classes
        self.data       = data
        self.batch_size = params.get('batch_size', 1024)
        self.n_pixel    = params['n_pixel']
        self.train_split      = params.get('train_split', 0.5)
        
        # Create image data in correct format
        self.unify_size()
        self.images     = self.trasform_to_images()
        self.labels, self.label2set     = self.generate_labels()
        data_store['label2set'] = self.label2set
        self.X = np.array(self.images).reshape(self.n_classes*self.N, 1, self.n_pixel, self.n_pixel)
        self.y = np.array(self.labels).reshape(self.n_classes*self.N)
        self.X = torch.from_numpy(self.X).to(self.device).to(torch.float32)
        self.y = torch.from_numpy(self.y).to(self.device).to(torch.long)
        self.X.requires_grad    = True 
        
        # Split in ([train,validation], [test])
        idx             = np.random.permutation(self.X.shape[0])
        self.N_train    = int(8/10 *self.train_split * self.N * self.n_classes)
        self.N_val      = int(2/10 *self.train_split * self.N * self.n_classes)
        self.N_test     = self.n_classes * self.N - self.N_train - self.N_val
        idx_train       = idx[:self.N_train] 
        idx_val         = idx[self.N_train:self.N_train+self.N_val]
        idx_test        = idx[self.N_train+self.N_val:]
        self.X_train, self.y_train    = self.X[idx_train], self.y[idx_train]
        self.X_val, self.y_val    = self.X[idx_val], self.y[idx_val]
        self.X_test, self.y_test      = self.X[idx_test], self.y[idx_test]
        
        data_store['set_sizes'] = {'N_train': self.N_train, 'N_val': self.N_val, 'N_test': self.N_test}
        logging.info(f'⤷ Datasplit: \n \
        Total / Train / Validation / Test \n \
        {self.N * self.n_classes} / {self.N_train} / {self.N_val} / {self.N_test}')
        
        self.save_test()
        


    def __call__(self, mode, idx):
        X, y = eval('self.X_'+mode), eval('self.y_'+mode)
        if idx == 0:
            self.permutation = np.random.permutation(y.shape[0])
        batch_idx   = self.permutation[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_X     = X[batch_idx]
        batch_y     = y[batch_idx]
        return batch_X, batch_y
    

    def n_batches(self, mode):
        N = eval('self.N_' + mode)
        N_batches = int(np.floor(N/self.batch_size))
        return N_batches


    def save_test(self):
        '''Saves the test data set for further use.'''
        logging.info('⤷ Saving test image data.')
        output_path = self.params['output_path']
        for set in self.label2set.items():
            os.makedirs(output_path + '/image_data/' + set[1], exist_ok=True) 
        for i in range(self.N_test):
            label   = self.y_test[i]
            set     = self.label2set[label.item()]
            np.save(output_path + '/image_data/' + set + '/X_test_{}'.format(i), self.X_test[i].detach().to(device='cpu').numpy())
            np.save(output_path + '/image_data/' + set + '/y_test_{}'.format(i), self.y_test[i].detach().to(device='cpu').numpy())

    def plot_test(self, model):
        logging.info('⤷ Plotting test images.')
        '''Plots some of the generated detector images.'''
        output_path = self.params['output_path']
        for set in self.label2set.items():
            os.makedirs(output_path + '/images/' + set[1], exist_ok=True) 

        n_pixel     = self.params.get('n_pixel', 10)
        eta_range   = self.params['eta_range']
        phi_range   = self.params['phi_range']
       
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        logging.info(f"{self.X_test.shape}")
        pred_test   = model(self.X_test).detach().cpu().numpy().squeeze()
        for i in range(self.N_test):
            label   = self.y_test[i]
            set     = self.label2set[label.item()]
            #logging.info(f"{self.X_test[i].shape}")
            #pred    = model(self.X_test[i].reshape(1,1,n_pixel,n_pixel)).detach().cpu().numpy().squeeze()
            pred    = pred_test[i]
            pred_string = ""
            for j in range(len(pred)):
                pred_string += self.label2set[j] + f":{round(pred[j],6)}_"
            #logging.info(f"{pred}")
            image   = self.X_test[i].detach().cpu().numpy().squeeze()
            plt.figure(figsize=(6, 6), dpi=160) 
            #image[image != 0] = np.log(image[image != 0])
            plt.imshow(image)
            plt.xlabel(r"$\eta$", fontsize=14)
            plt.ylabel(r"$\phi$", fontsize=14)
            plt.xticks(np.linspace(0,n_pixel-1,5), np.round(np.linspace(eta_range[0],eta_range[1], 5),1))
            plt.yticks(np.linspace(0,n_pixel-1,5), np.round(np.linspace(phi_range[0],phi_range[1], 5),1))
            plt.savefig(output_path + '/images/' + set + "/img_test_{}_".format(i) + pred_string + ".png", dpi=160, format="png")
            plt.close()
        

    def unify_size(self):
        '''Unifies the size of each class to the size of the smallest data set.'''
        sizes   = []
        for set in self.classes:
            sizes.append(self.data[set].shape[0])
        self.N  = np.min(sizes)         
        for set in self.classes:
            idx = np.arange(self.data[set].shape[0])
            self.data[set] = self.data[set][np.random.choice(idx, self.N),:,:]


    def trasform_to_images(self):
        '''Transforms the data to images.'''
        imgs        = []

        for set in self.classes:
            jets    = self.data[set]   

            if not set == "smileys":

                # Extract observables from jets array
                E       = jets[:,:,0]
                px      = jets[:,:,1]
                py      = jets[:,:,2]
                pz      = jets[:,:,3]

                # Compute (pt, eta, phi)-parametrization
                pt      = compute_pt(px, py)
                eta     = compute_eta(pt, pz)
                phi     = compute_phi(px, py)

                E_jet   = E.sum(axis=1)
                px_jet  = px.sum(axis=1)
                py_jet  = py.sum(axis=1)
                pz_jet  = pz.sum(axis=1)
                pt_jet  = compute_pt(px_jet, py_jet)
                m_jet   = compute_m(E_jet, px_jet, py_jet, pz_jet)
                #plt.hist(m_jet, bins=np.linspace(0,300,60))
                #plt.savefig("Mass"+set+".png", format="png")
                intensity = eval(self.params.get('intensity_measure', 'pt'))

                # Jet Preprocessing
                eta_preproc, phi_preproc = jet_preprocessing(eta, phi, intensity, self.params)

                # Make the Image
                images  = make_image(eta_preproc, phi_preproc, intensity, self.params)
            else:
                images  = jets

            # Image Preprocessing
            images_preproc = image_preprocessing(images, self.params)

            # Plot some images
            #plot_images(images_preproc, set, self.params)

            imgs.append(images_preproc)
            
        return imgs
        

    def generate_labels(self):
        '''Generates the labels for each class.'''
        labels      = []
        label2set   = {}
        for i in range(self.n_classes):
            set = self.classes[i]
            labels.append(i*np.ones(self.N))
            label2set[i]   = set
            
        return labels, label2set
        

class Rescale(object):
    '''Rescale the image to a totoal of 1.'''
    def __init__(self, norm=1):
        self.norm = 1

    def __call__(self, instance):
        return self.norm * instance / torch.sum(instance)

class Center(object):
    '''Center the given image.'''
    def __init__(self, n_pixel):
        self.n_pixel    = n_pixel

    def __call__(self, image):
        image       = np.array(image)

        # Parameters 
        n_pixel     = self.n_pixel
        center      = np.array([int(np.ceil(n_pixel/2)), \
                                int(np.ceil(n_pixel/2))])
        center_of_mass  = ndimage.measurements.center_of_mass(image)
        roll_values = ( center[0] - int(center_of_mass[0]), \
                        center[1] - int(center_of_mass[1]) ) 
        pad         = int(np.ceil(n_pixel/2))

        # pad and center the image
        image       = np.pad(image, ((pad,pad), (pad,pad)), mode="constant", constant_values=(0,0))
        image       = np.roll(image, (roll_values), axis=(0,1))

        # crop image back to original size
        x,y         = image.shape
        start_x     = x//2 - int(np.ceil(n_pixel/2))
        start_y     = y//2 - int(np.ceil(n_pixel/2))
        image       = image[start_x:start_x+n_pixel,\
                            start_y:start_y+n_pixel]
        return image
        
