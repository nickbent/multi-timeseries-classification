import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from layers import cnn1dblock, linear_layer, cnn1d
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

    def __init__(self, channels, kernel_sizes, sequence_length, 
                num_classes, attention = False, num_heads = 2,
                dropout = 0.8, lr = 0.001, betas = (0.9, 0.999), eps = 1e-8):
        super(MultiChannelBase, self).__init__()
        self.num_classes = num_classes
        self.attention = attention
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        
        self.conv = cnn1d(channels, kernel_sizes)

        if self.attention:
            self.num_final_layers = get_final_length(sequence_length, kernel_sizes)
            self.attention_layers = nn.MultiheadAttention(self.num_final_layers, num_heads)
        else:
            self.num_final_layers = channels[-1]*get_final_length(sequence_length, kernel_sizes)
        
        self.classifier = nn.Sequential(*linear_layer(self.num_final_layers, num_classes, drop_out = dropout))
        
        self.num_paramaters = sum(p.numel() for p in self.parameters())
        
    def forward(self, x):
        x = self.conv(x)
        if self.attention:
            x = torch.transpose(x, 0, 1)
            x, _ = self.attention_layers(x,x,x)
            x = torch.transpose(x, 0, 1)
            x = torch.mean(x, dim = 1)
        else:
            x = torch.flatten(x, start_dim = 1)
        pred = self.classifier(x)

        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas = self.betas, eps = self.eps)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        start = time.time()
        x, y = batch
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
        labels = y 
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, labels)
        self.test_acc(F.softmax(y_hat, dim =1), labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_accuracy', self.test_acc, on_step=True, on_epoch=True)
        self.log('test_batch_time', time.time()-start,  on_step=True, on_epoch=True)
        return {"test_loss" : loss}


class MultiChannelMultiTime(LightningModule):

    def __init__(self, channels_time, window_sizes, kernel_sizes_time, 
                num_classes, attention, num_heads = 2, dropout = 0.8, lr = 0.001, 
                betas = (0.9, 0.999), eps = 1e-8):
        super(MultiChannelMultiTime, self).__init__()
        self.num_classes = num_classes
        self.window_sizes = window_sizes
        self.num_times_scales = len(window_sizes)
        self.attention = attention
        self.lr = lr
        self. betas = betas
        self.eps = eps
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.conv = []

        for channels, kernels in zip( channels_time, kernel_sizes_time):
            conv = cnn1d(channels, kernels)
            if torch.cuda.is_available():
                self.conv.append(conv.cuda())
            else:
                self.conv.append(conv)

        if self.attention:
            self.num_final_layers = get_final_length(window_sizes[0], kernel_sizes_time[0])
            self.attention_layers = nn.MultiheadAttention(self.num_final_layers, num_heads)
        else:
            self.num_final_layers = sum([ channels[-1]*get_final_length(window_size, kernels) for window_size, kernels in zip(window_sizes, kernel_sizes_time)])


        self.classifier = nn.Sequential(*linear_layer(self.num_final_layers, num_classes, drop_out = dropout))
        
        self.num_paramaters = sum(p.numel() for p in self.parameters())
        
    def forward(self, x):
        cnn_out = []
        for conv, window_size in zip(self.conv,self.window_sizes):
            out= conv(x[:,:,:window_size])
            if not self.attention:
                out = torch.flatten(out, start_dim = 1)
            cnn_out.append(out)

        x = torch.cat(cnn_out, dim = 1)
        if self.attention:
            x = torch.transpose(x, 0, 1)
            x, _ = self.attention_layers(x,x,x)
            x = torch.transpose(x, 0, 1)
            x = torch.mean(x, dim = 1)

        pred = self.classifier(x)

        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas = self.betas, eps = self.eps)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        start = time.time()
        x, y = batch
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
        labels = y 
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, labels)
        self.test_acc(F.softmax(y_hat, dim =1), labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_accuracy', self.test_acc, on_step=True, on_epoch=True)
        self.log('test_batch_time', time.time()-start,  on_step=True, on_epoch=True)
        return {"test_loss" : loss}

class MultiChannelMultiTimeDownSample(LightningModule):

    def __init__(self, channels, window_sizes,
                down_sampling_kernel, kernel_sizes, 
                sequence_length, num_classes, attention = False, 
                num_heads = 2, dropout = 0.8, lr = 0.001, 
                betas = (0.9, 0.999), eps = 1e-8):
        super(MultiChannelMultiTimeDownSample, self).__init__()
        self.num_classes = num_classes
        self.window_sizes = window_sizes
        self.down_sampling_kernel = down_sampling_kernel
        self.num_times_scales = len(window_sizes)
        self.attention = attention
        self.lr = lr
        self. betas = betas
        self.eps = eps
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.conv = []

        for _ in range(self.num_times_scales):
            self.conv.append(cnn1d(channels, kernel_sizes))

        if self.attention:
            self.num_final_layers = get_final_length(sequence_length, kernel_sizes)
            self.attention_layers = nn.MultiheadAttention(self.num_final_layers, num_heads)
        else:
            self.num_final_layers = self.num_times_scales*channels[-1]*get_final_length(sequence_length, kernel_sizes)
        
        self.classifier = nn.Sequential(*linear_layer(self.num_final_layers, num_classes, drop_out = dropout))
        
        self.num_paramaters = sum(p.numel() for p in self.parameters())
        
    def forward(self, x):
        cnn_out = []
        for conv, kernels, window_size in zip(self.conv, self.down_sampling_kernel, self.window_sizes):
            x_window = x[:,:,:window_size]
            for kernel in kernels:
                x_window = torch.nn.functional.avg_pool1d(x_window, kernel, stride = 2)
            out=conv(x_window)
            if not self.attention:
                out = torch.flatten(out, start_dim = 1)
            cnn_out.append(out)

        x = torch.cat(cnn_out, dim = 1)
        if self.attention:
            x = torch.transpose(x, 0, 1)
            x, _ = self.attention_layers(x,x,x)
            x = torch.transpose(x, 0, 1)
            x = torch.mean(x, dim = 1)

        pred = self.classifier(x)

        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas = self.betas, eps = self.eps)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        start = time.time()
        x, y = batch
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
        labels = y 
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, labels)
        self.test_acc(F.softmax(y_hat, dim =1), labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_accuracy', self.test_acc, on_step=True, on_epoch=True)
        self.log('test_batch_time', time.time()-start,  on_step=True, on_epoch=True)
        return {"test_loss" : loss}

