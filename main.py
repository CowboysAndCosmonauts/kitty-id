
import numpy as np
from Network import Network
import h5py

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
    train_set_x, test_set_x = process_data(train_set_x_orig, test_set_x_orig)
    #4-layer model
    layer_sizes = [12288, 20, 7, 5, 1]

    N = Network(layer_sizes)
    model = N.train(train_set_x, train_set_y, print_cost=True)
    N.predict(train_set_x, train_set_y, model)
    N.predict(test_set_x, test_set_y, model)
