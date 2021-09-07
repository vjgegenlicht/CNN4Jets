import numpy as np
import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, n_classes, params, data_store):
        super(cnn, self).__init__()
        
        self.params     = params
        self.data_store = data_store
        self.n_classes  = n_classes
        
        self.batch_size = params.get('batchsize', 1024)
        
        self.loss       = nn.CrossEntropyLoss(reduction='mean')
        
        p_drop          = self.params.get('p_drop', 0.3)
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,4),
            nn.ReLU(),
            nn.Dropout2d(p=p_drop),
            nn.Conv2d(8,8,4),
            nn.ReLU(),
            nn.Dropout2d(p=p_drop),
            nn.MaxPool2d(4),
            nn.Conv2d(8,8,4),
            nn.ReLU(),
            nn.Dropout2d(p=p_drop),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,self.n_classes),
            nn.ReLU(),
            nn.Softmax(dim=1)         
        )
        
        self.set_optimizer()
        
        
    def forward(self, x):
        for layer in self.layers:  
            x = layer(x)
        return x
    
    
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
        ts_size     = Dataset.N_train 
        max_iter    = ts_size // self.batch_size
        
        for epoch in range(n_epochs):
            train_loss = 0
            for iter in range(max_iter):
                batch_X, batch_y = Dataset('train', iter)
                predict_y   = self(batch_X)
                
                self.optim.zero_grad()
                
                loss        = self.loss(predict_y, batch_y)
                
                loss.backward(retain_graph=True)
                self.optim.step()
                
                train_loss += loss.item()/max_iter
                
            val_loss = self.validate(Dataset)
            print(f'Epoch {epoch}\n \
                    Train Loss {train_loss}\n \
                    Validation Loss {val_loss}')
            
    def validate(self, Dataset):
        vs_size     = Dataset.N_val
        max_iter    = vs_size // self.batch_size
        val_loss    = 0
        
        for iter in range(max_iter):
            batch_X, batch_y = Dataset('val', iter)
            predict_y   = self(batch_X)
            loss        = self.loss(predict_y, batch_y)
            val_loss    += loss.item()/max_iter
        
        return val_loss
        
                
                
                
                