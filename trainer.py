from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from src.model import AutoEncoder
import pytorch_lightning as pl
from src.data import QuarkDataset, train_val_test_split  




decompressed__quark_path = "Data/processed/quark-gluon-uncompressed.hdf5"
cpu_count = mp.cpu_count()

dataset = QuarkDataset(decompressed__quark_path)
datasets = train_val_test_split(dataset)
train_loader = DataLoader(datasets['train'],  batch_size = 128, num_workers=cpu_count)
val_loader = DataLoader(datasets['val'],  batch_size = 128, num_workers=cpu_count)
test_loader = DataLoader(datasets['test'],  batch_size = 128, num_workers=cpu_count)

encoder_params = (3, 8, 5)
decoder_params = (5, 8, 3)
model = AutoEncoder(encoder_params, decoder_params)
trainer = pl.trainer.trainer.Trainer(max_epochs=10, accelerator="gpu", devices="auto")
trainer.fit(model, train_loader, val_loader)