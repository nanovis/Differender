import numpy as np
import pickle


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        return data


def dump_data(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_head_data():
    return load_data("./data/skewed_head.pickle")


def _load_head_data():
    data_path = "./data/skewed_head.dat"
    dims = np.fromfile(data_path, dtype=np.uint16, count=3)
    data = np.fromfile(data_path, dtype=np.uint16, offset=3 * 2) << 4
    data_3d = np.zeros(dims, dtype=np.uint16)
    for z in range(dims[2]):
        z_idx_base = z * dims[0] * dims[1]
        for y in range(dims[1]):
            y_idx_base = y * dims[0]
            for x in range(dims[0]):
                data_3d[x, y, z] = data[x + y_idx_base + z_idx_base]
    return dims, data_3d


if __name__ == "__main__":
    dims, data = _load_head_data()
    dump_data(data, "./data/skewed_head.pickle")
