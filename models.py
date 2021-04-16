import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from layers import cnn1dblock, linear_layer
from pytorch_lightning import LightningModule
import math
import time

def get_out_put_length(input_length, kernel, padding = 2, stride = 2):
    return math.floor(((input_length+2*padding-kernel)/stride +1))


def get_final_length(sequence_length, kernel_sizes, padding=2, stride=2):

    for kernel in kernel_sizes:
        sequence_length = get_out_put_length(sequence_length, kernel, padding, stride)
    return sequence_length


class MultiChannelBase(LightningModule):

    def __init__(self, channels, kernel_sizes, sequence_length, num_classes, dropout = 0.8, lr = 0.001, betas = (0.9, 0.999), eps = 1e-8):
        super(MultiChannelBase, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self. betas = betas
        self.eps = eps
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        
        conv = []

        for kernel in kernel_sizes:
            conv.append(cnn1dblock(channels, channels, kernel))
        
        self.conv = nn.Sequential(*conv)

        self.num_final_channels_flattened = channels*get_final_length(sequence_length, kernel_sizes)
        self.classifier = nn.Sequential(*linear_layer(self.num_final_channels_flattened, num_classes, drop_out = dropout))
        
        self.num_paramaters = sum(p.numel() for p in self.parameters())
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim = 1)
        #x = x.view(-1, self.num_final_channels_flattened)
        pred = self.classifier(x)

        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas = self.betas, eps = self.eps)
        #scheduler = cheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        start = time.time()
        x, y = batch
        #x = x.view(-1, 784, 1)
        labels = y 
        y_hat = self(x)
        train_loss = F.cross_entropy(y_hat, labels)
        self.train_acc(F.softmax(y_hat, dim =1), labels)
        self.log('train_loss_'+str(self.current_epoch), train_loss, on_step=False, on_epoch=True)
        self.log('train_accuracy_'+str(self.current_epoch), self.train_acc, on_step=False, on_epoch=True)
        self.log('train_batch_time_'+str(self.current_epoch), time.time()-start,  on_step=False, on_epoch=True)
        return {"loss" : train_loss}

    def validation_step(self, batch, batch_idx):
        start = time.time()
        x, y = batch
        #x = x.view(-1,  784, 1)
        labels = y 
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, labels)
        self.valid_acc(F.softmax(y_hat, dim =1), labels)
        self.log('valid_loss', val_loss, on_step=True, on_epoch=True)
        self.log('valid_accuracy', self.valid_acc, on_step=True, on_epoch=True)
        self.log('valid_batch_time', time.time()-start,  on_step=True, on_epoch=True)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        start = time.time()
        x, y = batch
        #x = x.view(-1,  784, 1)
        labels = y 
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, labels)
        self.test_acc(F.softmax(y_hat, dim =1), labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_accuracy', self.test_acc, on_step=True, on_epoch=True)
        self.log('test_batch_time', time.time()-start,  on_step=True, on_epoch=True)
        return {"test_loss" : loss}