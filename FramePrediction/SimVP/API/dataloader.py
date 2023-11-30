from .clevrer import load_clevrer

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    return load_clevrer(batch_size, val_batch_size, data_root, num_workers)
