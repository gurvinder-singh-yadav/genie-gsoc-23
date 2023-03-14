from torch.utils.data import IterableDataset, Dataset
from torch import from_numpy
import numpy as np
from itertools import cycle
import h5py
from torch import from_numpy

class IterableQuarkDataset(IterableDataset):
    def __init__(self, file_path, batch_size = 500) -> None:
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
    def parse_file(self):
        with h5py.File(self.file_path, 'r') as f:
            keys = list(f.keys())
            num_jets = 5000
            for i in range(0, num_jets, self.batch_size):
                X_jets = f[keys[0]][i:min(i+500, 1000)]
                for X_jet in X_jets:
                    X_jet = np.array([X_jet[..., i] for i in range(3)])
                    yield from_numpy(X_jet)
    def get_stream(self):
        return cycle(self.parse_file())
    def __iter__(self):
        return self.get_stream()
    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f["X_jets"])

class QuarkDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.file = h5py.File(path)
        self.keys = list(self.file.keys())

    def __len__(self):
        return len(self.file[self.keys[0]])
    def __getitem__(self, index):
        # data = [self.file[key][index] for key in self.keys]
        # data = [self.file[self.keys[0]][index], self.file[self.keys[-1]][index]]
        x = self.file['X_jets'][index]
        x = from_numpy(np.array([x[..., i] for i in range(3)]))
        return x
        