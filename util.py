import numpy as np
import torch
from sklearn.model_selection import train_test_split
from constit2images import *

def load_data(input_path):
    '''Load the data from the given path and transforms it to a 3d numpy array [n_jets, n_constituents, 4].
        args: 
            input_path:     : [string]  path to the txt file holding the data
    '''
    jets    = np.loadtxt(input_path)
    n_jets  = jets.shape[0]
    n_const = int(jets.shape[1]/4)
    jets    = jets.reshape(n_jets, n_const, 4)

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
        self.X = np.array(self.images).reshape(self.n_classes*self.N, 1, self.n_pixel, self.n_pixel)
        self.y = np.array(self.labels).reshape(self.n_classes*self.N)
        self.X = torch.from_numpy(self.X).to(self.device).to(torch.float32)
        self.y = torch.from_numpy(self.y).to(self.device).to(torch.long)
        self.X.requires_grad    = True 
        
        # Split in ([train,validation], [test])
        idx             = np.random.permutation(self.X.shape[0])
        self.N_train    = int(9/10 *self.train_split * self.N * self.n_classes)
        self.N_val      = int(1/10 *self.train_split * self.N * self.n_classes)
        self.N_test     = self.N - self.N_train - self.N_val
        idx_train       = idx[:self.N_train] 
        idx_val         = idx[self.N_train:self.N_train+self.N_val]
        idx_test        = idx[self.N_train+self.N_val:]
        self.X_train, self.y_train    = self.X[idx_train], self.y[idx_train]
        self.X_val, self.y_val    = self.X[idx_val], self.y[idx_val]
        self.X_test, self.y_test      = self.X[idx_test], self.y[idx_test]
        print(self.y_train.shape, self.y_val.shape, self.y_test.shape)
        
    def __call__(self, mode, idx):
        X, y = eval('self.X_'+mode), eval('self.y_'+mode)
        if idx == 0:
            self.permutation = np.random.permutation(y.shape[0])
        batch_idx   = self.permutation[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_X     = X[batch_idx]
        batch_y     = y[batch_idx]
        return batch_X, batch_y
     
            
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

            # Extract observables from jets array
            E       = jets[:,:,1]
            px      = jets[:,:,0]
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
            intensity = eval(self.params.get('intensity_measure', 'pt'))

            # Jet Preprocessing
            eta_preproc, phi_preproc = jet_preprocessing(eta, phi, intensity, self.params)

            # Make the Image
            images = make_image(eta_preproc, phi_preproc, intensity, self.params)

            # Image Preprocessing
            images_preproc = image_preprocessing(images, self.params)

            # Plot some images
            plot_images(images_preproc, set, self.params)

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
        
        
        