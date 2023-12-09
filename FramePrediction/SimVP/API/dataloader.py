from .clevrer import load_clevrer, load_clevrer_inference_data

def load_data(dataname, batch_size, val_batch_size, data_root, num_workers, **kwargs):
    is_inference = kwargs.get("inference", False)
    if is_inference:
        inference_data_root = kwargs["inference_data_root"]
        return load_clevrer_inference_data(batch_size, inference_data_root, num_workers)
    # print(batch_size, val_batch_size, data_root, num_workers)
    return load_clevrer(batch_size, val_batch_size, data_root, num_workers)
