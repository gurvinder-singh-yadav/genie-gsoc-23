import h5py
import os.path as osp
def subset_dataset(raw_path, processed_dir, subset_len, starter = 0):
    processed_path = osp.join(processed_dir, "data.hdf5")
    with h5py.File(raw_path, 'r') as f, h5py.File(processed_path, 'w') as p:
        keys = list(f.keys())
        total_events = f[keys[1]].shape[0]
        for key in keys:
            shape = (subset_len,)
            if len(f[key].shape) > 1:
                shape = (subset_len, 125, 125, 3)
            p.create_dataset(key, shape=shape)
        quark_count = 0
        gluon_count = 0
        idx = 0
        j = 0
        for i in range(starter, total_events):
            if quark_count < subset_len // 2:
                for key in keys:
                    p[key][idx] = f[key][i]
                quark_count += 1
                # print(idx, i)
                idx += 1
                j = i
            elif gluon_count < subset_len // 2:
                for key in keys:
                    p[key][idx] = f[key][i]
                # print(idx, i)
                gluon_count += 1
                idx+=1
                j = i
        return j + starter


train_path = "Data/Graphs/train/raw"
quark_gluon_path = "Data/hdf5/processed/processed.hdf5"
val_path = "Data/Graphs/val/raw"
test_path = "Data/Graphs/test/raw"

i = subset_dataset(quark_gluon_path, train_path, 6000, 0)
print(i)    
i = subset_dataset(quark_gluon_path, val_path, 1200, i)
print(i)    
i = subset_dataset(quark_gluon_path, test_path, 1200, i)
print(i)   