import numpy as np


def load_head_data():
    data_path = "./data/skewed_head.dat"
    dims = np.fromfile(data_path, dtype=np.uint16, count=3)
    data = np.fromfile(data_path, dtype=np.uint16, offset=3 * 2)
    data = data << 4
    return dims, data.reshape(dims)


if __name__ == "__main__":
    dims, data = load_head_data()
    fdata = data / float(np.iinfo(np.uint16).max)
    print(dims)
    print(data.shape)
    print(fdata.dtype)
