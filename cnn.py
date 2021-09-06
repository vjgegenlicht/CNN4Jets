import numpy as np
import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, n_classes, params, data_store):
        super(cnn, self).__init__()
        
        self.params     = params
        self.data_store = data_store
        self.n_classes  = n_classes
        
        self.loss       = nn.CrossEntropyLoss
        
        self.layers = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )
        
        self.set_optimizer()
        
        
    def forward(self, x):
        out     = self.layers(x)
        return out
    
    
    def set_optimizer(self):
        Parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        
        self.optim  = torch.optim.AdamW(
            Parameters,
            self.params.get('lr', 1.e-4),
            betas = self.params.get('betas', [0.9,0.99]),
            weight_decay = self.params.get('L2', 0.0)
        )
    
    
    def train(self, Dataset):
        
        n_epochs    = self.params['n_epochs']
        batch_size  = self.params.get('batch_size', 1024)
        ts_size     = Dataset.N 
        max_iter    = ts_size // batch_size
        
        for epoch in range(n_epochs):
            print(epoch)
            for iter in range(max_iter):
                batch_X, batch_y = Dataset(iter)
                predict_y   = self(batch_X)
                                
                optimizer.zero_grad()
                
                loss        = self.loss(predict_y, batch_y, reduction='mean')
                
                loss.backward()
                optimizer.step()
                
                