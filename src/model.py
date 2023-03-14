from torch import nn 
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
    
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ## encoding layers
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Conv2d(in_channels=9, out_channels=14, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Conv2d(in_channels=14, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
            
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=3, stride=2),
            nn.ConvTranspose2d(in_channels=4, out_channels=14, kernel_size=2, stride=2, output_padding=1),
            nn.ConvTranspose2d(in_channels=14, out_channels=9, kernel_size=3, stride=2),
            nn.ConvTranspose2d(in_channels=9, out_channels=3, kernel_size=1, stride=2)
        )
    def forward(self, x):
        reconstructed = self.decoder(x)
        return reconstructed
    

class AutoEncoder(pl.LightningModule):
    def __init__(self,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    def _get_reconstruction_loss(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction='none')
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
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
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
        return loss
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
        return loss