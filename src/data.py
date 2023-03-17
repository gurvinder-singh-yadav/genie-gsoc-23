from torch.utils.data import IterableDataset, Dataset, random_split
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
        
def uncompress_dataset(raw_path,  processed_path):
    with h5py.File(raw_path, 'r') as r, h5py.File(processed_path, 'w') as p:
        keys = list(r.keys())
        total_events = r[keys[0]].shape[0]
        for key in keys:
            if len(r[key].shape) > 1:
                chunk_shape = tuple([6000] + list(r[key].shape[1:]))
            else:
                chunk_shape = (6000,)
            p.create_dataset(key, shape=r[key].shape, chunks= chunk_shape)
            for i in range(0, total_events, 6000):
                stop_idx = min(i+6000, total_events)
                p[key][i:stop_idx] = r[key][i:stop_idx]

def train_val_test_split(dataset, train = 0.7, val = 0.1, test = 0.2):
    train_data, val_data, test_data = random_split(dataset, [0.6, 0.2, 0.2])
    datasets = {}
    datasets['train'] = train_data
    datasets['val'] = val_data
    datasets['test'] = test_data
    return datasets