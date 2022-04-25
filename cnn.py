import numpy as np
import torch
import torch.nn as nn
import os
import logging
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as tvtrfs
from torch.utils.data import DataLoader
from util import Rescale
from torch.nn.modules import conv

class cnn(nn.Module):
    def __init__(self, params, data_store):
        super(cnn, self).__init__()
        
        self.params             = params
        self.data_store         = data_store
        self.device             = data_store["device"]
        self.n_classes          = params["n_classes"]
        #logging.info(f"Classes: {self.n_classes}")
        self.lr_scheduler_mode  = self.params.get("lr_scheduler", "one_cycle_lr")

        self.forward_passes     = 0

        self.batch_size = params.get('batch_size', 1024)
        
        self.loss       = nn.CrossEntropyLoss(reduction='mean')
        
        self.layers = nn.Sequential(
            ### first architecture ###
            self.convolution(1,8,4),
            nn.BatchNorm2d(8),
            self.convolution(8,8,4),
            nn.MaxPool2d(2),
            self.convolution(8,8,4),
            nn.MaxPool2d(2),
            nn.Flatten(),
            self.linear(648,32),
            nn.BatchNorm1d(32),
            self.linear(32,32),
            self.linear(32,32),
            self.linear(32,self.n_classes),
            nn.Softmax(dim=1)

            ### Other Architecture ###
            #self.convolution(1,128,4),
            #self.convolution(128,64,4),
            #nn.MaxPool2d(2),
            #self.convolution(64,64,4),
            #self.convolution(64,64,4),
            #nn.MaxPool2d(2),
            #nn.Flatten(),
            #self.linear(4096,64),
            #self.linear(64,256),
            #self.linear(256,256),
            #self.linear(256,256),
            #self.linear(256, self.n_classes),
            #nn.Softmax(dim=1)         
        )
        
    def convolution(self, in_channels, out_channels, size):
        p_drop          = self.params.get('p_drop', 0.3)
        convolution     = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, size, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=p_drop)
        )
        return convolution
    
    def linear(self, in_nodes, out_nodes):
        linear          = nn.Sequential(
            nn.Linear(in_nodes, out_nodes),
            nn.ReLU(),
        )
        return linear
    
        
    def forward(self, x):
        i = 0 
        #if self.forward_passes == 0:
            #logging.info("Shape of the model:")
        for layer in self.layers:  
            x = layer(x)
   
            #if self.forward_passes == 0:
                #logging.info(f"{x.shape}")
            if self.params.get('output_layers', False):
                output_path = self.params['output_path']
                os.makedirs(output_path + '/layers/', exist_ok=True)
                np.save(output_path + '/layers/' + 'layer_{}.npy'.format(i), x.detach().to(device='cpu').numpy())
            i += 1
        self.forward_passes += 1
        return x
    
    def initialize_dataloaders(self):
        transforms  = tvtrfs.Compose([tvtrfs.Grayscale(num_output_channels=1), tvtrfs.ToTensor(), Rescale(norm=1)])
        train_dataset           = ImageFolder("./Dataset/train", transform=transforms)
        val_dataset             = ImageFolder("./Dataset/val", transform=transforms)
        test_dataset            = ImageFolder("./Dataset/test", transform=transforms)
        self.train_dataloader   = DataLoader(train_dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        self.val_dataloader     = DataLoader(val_dataset, 
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            drop_last=True)
        self.test_dataloader    = DataLoader(test_dataset, 
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            drop_last=True)
        self.data_store["label2set"]    = train_dataset.class_to_idx
        

    def set_optimizer(self):
        Parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.optim  = torch.optim.AdamW(
            Parameters,
            self.params.get('lr', 1.e-4),
            betas = self.params.get('betas', [0.9,0.99]),
            weight_decay = self.params.get('L2', 0.0)
        )


    def set_scheduler(self):
        if self.lr_scheduler_mode == "one_cycle_lr":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                10 * self.params.get('lr', 1.e-4),
                epochs = self.params['n_epochs'],
                steps_per_epoch = len(self.train_dataloader)
            )
        elif self.lr_scheduler_mode == "reduce_on_plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                mode="min",
                factor = 0.1,
            )
    
    
    def train_model(self):
        n_epochs    = self.params['n_epochs']
        max_iter    = len(self.train_dataloader)
        logging.info(f"Epochs: {n_epochs}, max_iter: {max_iter}")
        self.epoch  = 0
        for epoch in range(n_epochs):
            self.epoch += 1
            train_loss = 0
            for i in range(max_iter):
                batch_X, batch_y = next(iter(self.train_dataloader))
                batch_X     = batch_X.to(self.device)
                batch_y     = batch_y.to(self.device)
                predict_y   = self(batch_X)
                
                self.optim.zero_grad()
                
                loss        = self.loss(predict_y, batch_y)
                
                loss.backward(retain_graph=True)
                self.optim.step()
                
                train_loss += loss.item()/max_iter
                
                if self.lr_scheduler_mode == "one_cycle_lr":
                    self.lr_scheduler.step()
            
            # Validate
            val_loss, top_1_error = self.validate()
            if self.lr_scheduler_mode == "reduce_on_plateau":
                self.lr_scheduler.step(val_loss)
            
            # Outprint
            lr = np.round(self.lr_scheduler.optimizer.param_groups[0]['lr'], 4)
            logging.info(f'⤷ Epoch {epoch}\n Learning rate {lr}\n Train loss {train_loss}\n Validation loss {val_loss}\n Top one error {top_1_error * 100} %')
                     
            
    def validate(self):
        max_iter    = len(self.val_dataloader)
        val_loss    = 0
        top_1_error = 0
        
        for i in range(max_iter):
            with torch.no_grad():
                batch_X, batch_y = next(iter(self.val_dataloader))
                batch_X     = batch_X.to(self.device)
                batch_y     = batch_y.to(self.device)
                predict_y   = self(batch_X)

                loss        = self.loss(predict_y, batch_y)
                val_loss    += loss.item() / max_iter
                top_1_error += np.mean(np.where(np.argmax(predict_y.detach().to(device='cpu').numpy(), axis=1) != batch_y.detach().to(device='cpu').numpy(), True, False)) / max_iter

        return val_loss, top_1_error
    
    def check_calibration(self, Dataset):
        X_test  = Dataset.X_test
        y_test  = Dataset.y_test.detach().to(device='cpu').numpy()
        size    = Dataset.N_test
        n_pixel = self.params.get('n_pixel', 10)#
        batch_size  = self.params["batch_size"]
        max_iter    = size // batch_size
        steps   = np.linspace(0,1, 11)
        cal     = np.zeros_like(steps)
        for i in range(max_iter):
            sig = self(X_test.reshape(size,1,n_pixel,n_pixel)[i*batch_size: (i+1)*batch_size, :, :, :]).detach().cpu().numpy().squeeze()
            y_batch = y_test[i*batch_size : (i+1)*batch_size]
            y_pred  = np.argmax(sig, axis=1)
            prob    = np.max(sig, axis=1)
            steps   = np.linspace(0,1, 11)
            cal_batch   = []
            for j in range(len(steps)):
                step    = steps[j]
                mask    = ((prob > step-0.05) & (prob < step+0.05)).squeeze()
                cal_batch_step = ( np.mean( np.where(y_pred[mask] == y_batch[mask], True, False) ) / max_iter )
                cal[j]     += cal_batch_step
        logging.info(f"{steps}, {cal}")
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

        plt.figure(figsize=(10, 6), dpi=160)
        plt.plot(steps, cal)
        plt.xlabel("Certainty of the classification")
        plt.ylabel("Percentage of correctly classified instances")
        plt.savefig(self.params['output_path'] + "calibration.png", dpi=160, format="png")

    
    def predict_instance(self, instance):
        with torch.no_grad():
            instance    = instance.to(self.device)
            prob        = self(instance)
        return prob
    
    
    def predict_test(self, Dataset):
        max_iter    = len(self.train_dataloader)

        predict_prob   = []
        true_y         = []
        
        for iter in range(max_iter):
            batch_X, batch_y = next(iter(self.train_dataloader))
            predict_prob.append(self(batch_X).detach().to(device='cpu').numpy()) 
            true_y.append(batch_y.detach().to(device='cpu').numpy())
        
        true_y          = np.array(true_y).flatten()
        predict_prob    = np.array(predict_prob).reshape(max_iter * self.batch_size, self.n_classes)
        predict_y       = np.argmax(predict_prob, axis=1)
        
        output_path = self.params['output_path']
        os.makedirs(output_path, exist_ok=True)
        np.save(output_path + '/test_true_y', true_y)
        np.save(output_path + '/test_predict_prob', predict_prob)
        np.save(output_path + '/test_predict_y', predict_y)
        
        return true_y, predict_prob, predict_y

    def plot_test(self):
        logging.info('⤷ Plotting test images.')
        '''Plots some of the generated detector images.'''
        output_path = self.params['output_path']
        for set in list(self.data_store["label2set"].keys()):
            os.makedirs(output_path + '/images/' + set, exist_ok=True) 

        n_pixel     = self.params.get('n_pixel', 10)
        eta_range   = self.params['eta_range']
        phi_range   = self.params['phi_range']
       
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        
        for i in range(len(self.test_dataloader)):
            batch_X, batch_y    = next(iter(self.test_dataloader))
            batch_X             = batch_X.to(self.device) 
            with torch.no_grad():
                pred_test   = self(batch_X).detach().cpu().numpy().squeeze()
            for i in range(len(pred_test)):
                label   = batch_y[i]
                set     = list(self.data_store["label2set"].keys())[label.item()]

                pred    = pred_test[i]
                pred_string = ""
                for j in range(len(pred)):
                    pred_string += list(self.data_store["label2set"].keys())[j] + f":{round(pred[j],6)}_"
                image   = batch_X[i].detach().cpu().numpy().squeeze()
            
                plt.imsave(output_path + '/images/' + set + "/img_test_{}_".format(i) + pred_string + ".png", image, cmap="gray")
    
    def save(self):
        output_path = self.params['output_path']
        os.makedirs(output_path, exist_ok=True)
        torch.save({'optim': self.optim.state_dict(),
                    'data_loaders': {'train': self.train_dataloader,
                                     'val': self.val_dataloader,
                                     'test': self.test_dataloader},
                    'net': self.state_dict(),
                    'epoch': self.epoch,
                    'label2set': self.data_store['label2set']}, output_path+'/model')
    
    def load(self, path, load_optim=True, load_dataloaders=True):
        state_dicts = torch.load(path)
        self.load_state_dict(state_dicts['net'])
        if load_optim:
            self.optim.load_state_dict(state_dicts['optim'])
        
        self.epoch                      = state_dicts['epoch']      
        self.data_store['label2set']    = state_dicts['label2set']
        if load_dataloaders:
            self.train_dataloader           = state_dicts['data_loaders']['train']
            self.val_dataloader             = state_dicts['data_loaders']['val']
            self.test_dataloader            = state_dicts['data_loaders']['test']
                
                
                
