from torch.utils.data import IterableDataset, Dataset, random_split
from torch import from_numpy
import numpy as np
from itertools import cycle
import h5py
import networkx as nx

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

class QuarkDatasetTask1(Dataset):
    def __init__(self, path, channels = 3) -> None:
        super().__init__()
        self.path = path
        self.channels = channels
        self.file = h5py.File(path)
        self.keys = list(self.file.keys())

    def __len__(self):
        return len(self.file[self.keys[0]])
    def __getitem__(self, index):
        x = self.file['X_jets'][index]
        x = from_numpy(np.array([x[..., i] for i in range(self.channels)]))
        # print(x.shape)
        return x
    
class QuarkDatasetTask2(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
    
        
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


class ImageToGraph:
    def __init__(self, img,m0, pt, y, idx, k = 4) -> None:
        """ 
        input -> image : shape(m,n,3)
        output ->  Class object
        """
        self.k = k
        self.img = img
        self.idx = idx
        self.m0 = m0
        self.pt = pt
        self.y = y
        self.channels = 3   # total channels in the image
        self.point_clouds = {}
        self.graphs = {}
        self.channel_spliter() 
        self.image_to_point_cloud()
        self.create_graphs()
        self.write_graphs(self.idx)
    def channel_spliter(self):
        """ convert (m,n,3) -> (3,m,n) """
        channel = 3
        self.img = np.array([self.img[..., i] for i in range(channel)])
    def get_addrs(self, img):
        """Return: indices of all non zero-point in a particular channel of image"""
        return np.argwhere(img != 0)
    def get_val(self, img):
        """Return: all non-zero values in a particular channel of image """
        return img[img != 0 ]
    def get_z(self, addr):
        """Return:  3d representation formed from a 2d image of a channel
        ref: http://www.open3d.org/docs/release/tutorial/geometry/working_with_numpy.html
        """
        x_2 = addr.T[0]**2
        y_2 = addr.T[1]**2
        den = x_2 + y_2
        num = np.sin(x_2+y_2)
        num[den == 0] = 0
        den[den == 0] = 1
        return num/den
    def image_to_point_cloud(self):
        """Returns:  Dict[point_cloud] -> collection of point clouds
            point_cloud = {(n,3), (n,)}
            (n,3) -> 3d coordinated of that channel
            (n,1) -> value of that particular channel at its 2d index
        """
        for i in range(self.channels):
            addrs = self.get_addrs(self.img[i])
            vals = self.get_val(self.img[i])
            z = self.get_z(addrs).reshape(-1, 1)
            xyz = np.hstack((addrs, z))
            self.point_clouds[i] = {"3dVector":xyz, "values":vals}
    def get_k_nearest(self, curr_pos, pos, k = 4):
        """ 
        Returns: k closest neighbours of the node
        """
        dist = np.sum(pos-curr_pos, axis=1)
        k_closest = np.argsort(dist)[:k]
        return k_closest
    def get_edge_list(self, channel):
        """ 
        Returns: edgelist of the graph of that channel
        """
        point_cloud = self.point_clouds[channel]
        pos = point_cloud['3dVector']
        edge_list = []
        for i in range(len(pos)):
            dist = self.get_k_nearest(pos[i], pos, self.k)
            edge_list.extend(list(zip([i]*4, dist)))
        return edge_list
    def get_nodes_attrs(self, channel):
        """  
        Returns: attribute of every unique node
        """
        img = self.img[channel]
        addrs = self.get_addrs(img)
        vals = self.get_val(img).reshape(-1, 1)
        attrs = np.hstack((addrs, vals)).astype(np.float32)
        return attrs
    def create_graphs(self):
        """  
        Creates Graphs for every channel using networkx
        """
        for i in range(self.channels):
            attrs = self.get_nodes_attrs(i)
            edge_list = self.get_edge_list(i)
            G = nx.Graph()
            G.add_edges_from(edge_list)
            attributes = {}
            for j in range(len(G.nodes())):
                attributes[j] = {
                    "x": attrs[j][0],
                    "y": attrs[j][1],
                    "val": attrs[j][2],
                    "m0": self.m0,
                    "pt": self.pt,
                    "y":self.y
                }
                nx.set_node_attributes(G, attributes)
            self.graphs[i] = G

    def write_graph(self,channel, idx):
        nx.write_graphml(self.graphs[0], "Data/raw/Graphs/{}-{}.graphml".format(channel, idx))
    def read_graph(self, channel, idx):
        graph = nx.read_graphml("Data/raw/Graphs/{}-{}.graphml".format(channel, idx))
        return graph
    def write_graphs(self, idx):
        for i in range(self.channels):
            nx.write_graphml(self.graphs[i], "Data/raw/Graphs/{}-{}.graphml".format(i, idx))