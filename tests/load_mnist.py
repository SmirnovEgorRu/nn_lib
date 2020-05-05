import os
import requests
import numpy as np
import pandas as pd


def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=2**20):
                if chunk:
                    f.write(chunk)
    return local_filename


def load_mnist(dtype=np.float32):
    if not os.path.isfile("mnist.npz"):
        print("Loading data set...")
        download_file("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
    print("Reading data set...")
    with np.load("mnist.npz") as f:
        train_label = np.ascontiguousarray(f['y_train'], dtype)
        train_data = np.ascontiguousarray(f['x_train'].reshape((60000, 28*28)), dtype)
        test_label = np.ascontiguousarray(f['y_test'], dtype)
        test_data = np.ascontiguousarray(f['x_test'].reshape((10000, 28*28)), dtype)
    n_classes = len(np.unique(train_label))
    n_features = train_data.shape[1]

    return train_data, train_label, test_label, test_data, n_classes, n_features


if __name__ == '__main__':
    train_data, train_label, test_label, test_data, _, _ = load_mnist()

    train = pd.DataFrame(train_data)
    train["y"] = train_label

    test = pd.DataFrame(test_data)
    test["y"] = test_label

    newpath = r'data'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    train.to_csv(newpath + "/train.csv", header=False)
    test.to_csv(newpath + "/test.csv", header=False)
