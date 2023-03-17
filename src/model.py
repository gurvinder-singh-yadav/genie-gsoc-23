from torch import nn 
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
    
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn as nn, optim
import torch.nn.functional as F
import multiprocessing as mp
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, in_dim = 3, latent_dim = 6, out_dim = 3, kernel = 3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.kernel = kernel
        self.encoder = nn.Sequential(
            ## encoding layers
            # PrintDim(),
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.latent_dim * 3, kernel_size=self.kernel, stride=1, padding=1),
            # PrintDim(),
            nn.MaxPool2d(kernel_size=2, padding=1),
            # PrintDim(),
            nn.Conv2d(in_channels=self.latent_dim * 3, out_channels=self.latent_dim * 2, kernel_size=self.kernel, padding=1),
            # PrintDim(),
            nn.MaxPool2d(kernel_size=2, padding=1),
            # PrintDim(),
            nn.Conv2d(in_channels=self.latent_dim * 2, out_channels=self.latent_dim, kernel_size=self.kernel, stride=1, padding=1),
            # PrintDim(),
            nn.MaxPool2d(kernel_size=2, padding=1),
            # PrintDim(),
            nn.Conv2d(in_channels=self.latent_dim, out_channels=self.out_dim, kernel_size=self.kernel, stride=1, padding=1)
            
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class PrintDim(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        print(x.shape)
        print("-" * 50)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_dim = 3, latent_dim = 6, out_dim = 3, kernel = 3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.kernel = kernel
        self.decoder = nn.Sequential(
            # PrintDim(),
            nn.ConvTranspose2d(in_channels=self.in_dim, out_channels=self.latent_dim * 3, kernel_size=self.kernel, stride=2, padding=1),
            # PrintDim(),
            nn.ConvTranspose2d(in_channels=self.latent_dim * 3, out_channels=self.latent_dim * 2, kernel_size=self.kernel, stride=2, padding=2),
            # PrintDim(),
            nn.ConvTranspose2d(in_channels=self.latent_dim * 2, out_channels=self.out_dim, kernel_size=self.kernel, stride=2, padding=1),
            # PrintDim(),
        )
    def forward(self, x):
        reconstructed = self.decoder(x)
        return reconstructed
    
class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder_params, decoder_params):
        super().__init__()
        self.encoder = Encoder(*encoder_params)
        self.decoder = Decoder(*decoder_params)
        self.save_hyperparameters()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    def _get_reconstruction_loss(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction='none')
        # loss = loss.sum().mean()
        return loss
    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr = 1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5
                                                         )
        return {"optimizer": optimiser, "lr_scheduler": scheduler, "monitor":"val_loss"}
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss.item()
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
        return loss.item()
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
        return loss.item()