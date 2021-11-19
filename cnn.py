import numpy as np
import torch
import torch.nn as nn
import os
import logging

from torch.nn.modules import conv

class cnn(nn.Module):
    def __init__(self, n_classes, params, data_store):
        super(cnn, self).__init__()
        
        self.params             = params
        self.data_store         = data_store
        self.device             = data_store["device"]
        self.n_classes          = n_classes
        logging.info(f"Classes: {self.n_classes}")
        self.lr_scheduler_mode  = self.params.get("lr_scheduler", "one_cycle_lr")

        self.forward_passes     = 0

        self.batch_size = params.get('batch_size', 1024)
        
        self.loss       = nn.CrossEntropyLoss(reduction='mean')
        
        self.layers = nn.Sequential(
            self.convolution(1,8,4),
            self.convolution(8,8,4),
            nn.MaxPool2d(2),
            #nn.MaxPool2d(4),
            self.convolution(8,8,4),
            nn.MaxPool2d(2),
            nn.Flatten(),
            self.linear(648,32),
            #self.linear(128,32),
            ####self.linear(192,32),
            ###self.linear(972,32),
            self.linear(32,32),
            self.linear(32,32),
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
        if self.forward_passes == 0:
            logging.info("Shape of the model:")
        for layer in self.layers:  
            x = layer(x)

            if self.forward_passes == 0:
                logging.info(f"{x.shape}")
            if self.params.get('output_layers', False):
                output_path = self.params['output_path']
                os.makedirs(output_path + '/layers/', exist_ok=True)
                np.save(output_path + '/layers/' + 'layer_{}.npy'.format(i), x.detach().to(device='cpu').numpy())
            i += 1
        self.forward_passes += 1
        return x
    
    
    def set_optimizer(self):
        Parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        
        self.optim  = torch.optim.AdamW(
            Parameters,
            self.params.get('lr', 1.e-4),
            betas = self.params.get('betas', [0.9,0.99]),
            weight_decay = self.params.get('L2', 0.0)
        )


    def set_scheduler(self, Dataset):
        
        n_train_batches = Dataset.n_batches('train')
        if self.lr_scheduler_mode == "one_cycle_lr":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                10 * self.params.get('lr', 1.e-4),
                epochs = self.params['n_epochs'],
                steps_per_epoch = n_train_batches
            )
        elif self.lr_scheduler_mode == "reduce_on_plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                mode="min",
                factor = 0.1,
            )
    
    
    def train(self, Dataset):
        
        n_epochs    = self.params['n_epochs']
        ts_size     = Dataset.N_train 
        max_iter    = ts_size // self.batch_size
        logging.info(f"Epochs: {n_epochs}, max_iter: {max_iter}")
        self.epoch  = 0
        for epoch in range(n_epochs):
            self.epoch += 1
            train_loss = 0
            for iter in range(max_iter):
                batch_X, batch_y = Dataset('train', iter)
                predict_y   = self(batch_X)
                
                self.optim.zero_grad()
                
                loss        = self.loss(predict_y, batch_y)
                
                loss.backward(retain_graph=True)
                self.optim.step()
                
                train_loss += loss.item()/max_iter
                
                if self.lr_scheduler_mode == "one_cycle_lr":
                    self.lr_scheduler.step()
            
            # Validate
            val_loss, top_1_error = self.validate(Dataset)
            if self.lr_scheduler_mode == "reduce_on_plateau":
                self.lr_scheduler.step(val_loss)
            
            # Outprint
            lr = np.round(self.lr_scheduler.optimizer.param_groups[0]['lr'], 4)
            logging.info(f'⤷ Epoch {epoch}\n \
        Learning rate {lr}\n \
        Train loss {train_loss}\n \
        Validation loss {val_loss}\n \
        Top one error {top_1_error * 100} %')
                     
            
    def validate(self, Dataset):
        vs_size     = Dataset.N_val
        max_iter    = vs_size // self.batch_size
        val_loss    = 0
        top_1_error = 0
        
        for iter in range(max_iter):
            batch_X, batch_y = Dataset('val', iter)
            predict_y   = self(batch_X)
            loss        = self.loss(predict_y, batch_y)
            val_loss    += loss.item() / max_iter
            top_1_error += np.mean(np.where(np.argmax(predict_y.detach().to(device='cpu').numpy(), axis=1) != batch_y.detach().to(device='cpu').numpy(), True, False)) / max_iter
        return val_loss, top_1_error
    
    
    def predict_instance(self, instance):
        prob = self(instance)
        return prob
    
    
    def predict_test(self, Dataset):
        ts_size     = Dataset.N_test
        max_iter    = ts_size // self.batch_size

        predict_prob   = []
        true_y         = []
        
        for iter in range(max_iter):
            batch_X, batch_y = Dataset('test', iter)
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
    
    def save(self):
        output_path = self.params['output_path']
        os.makedirs(output_path, exist_ok=True)
        torch.save({'optim': self.optim.state_dict(),
                    'net': self.state_dict(),
                    'epoch': self.epoch,
                    'label2set': self.data_store['label2set']}, output_path+'/model')
    
    def load(self, path, load_optim=True):
        state_dicts = torch.load(path)
        self.load_state_dict(state_dicts['net'])
        if load_optim:
            self.optim.load_state_dict(state_dicts['optim'])
        self.epoch = state_dicts['epoch']      
        self.data_store['label2set'] = state_dicts['label2set']
                
                
                