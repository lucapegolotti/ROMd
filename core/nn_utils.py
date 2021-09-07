import tensorflow as tf
import numpy as np
from io_utils import save_junctions_data, save_sequential_data

def train_network(X, Y):
    batchsize = 20

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer = 'adam',
       loss = tf.keras.losses.MeanSquaredError(reduction = "auto", name = "mean_squared_error"),
       metrics = ['mae'])

    model.fit(X, Y, epochs = 100, validation_split = 0.0, batch_size = batchsize)

    return model

def train_and_save_all_networks(dataset, max_stencil_size):

    half = int(np.floor((max_stencil_size - 1) / 2))

    njunctions = [j for j in dataset.junctions_datasets]

    for njun in njunctions:
        var = 'pressure'
        center = half
        X, Y, mins, maxs, my, My = dataset.get_junction_dataset(njun, max_stencil_size, var)
        model = train_network(X, Y)
        save_junctions_data(X, Y, mins, maxs, my, My, model, max_stencil_size, njun, var)

        var = 'velocity'
        center = half
        X, Y, mins, maxs, my, My = dataset.get_junction_dataset(njun, max_stencil_size, var)
        model = train_network(X, Y)
        save_junctions_data(X, Y, mins, maxs, my, My, model, max_stencil_size, njun, var)

    for stencil_size in range(half + 1, max_stencil_size + 1):
        # pressure
        if stencil_size != max_stencil_size:
            var = 'pressure'
            center = half
            X, Y, mins, maxs, my, My = dataset.get_sequential_dataset(stencil_size, center, var)
            model = train_network(X, Y)
            save_sequential_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

            var = 'pressure'
            center = stencil_size - half - 1
            X, Y, mins, maxs, my, My = dataset.get_sequential_dataset(stencil_size, center, var)
            model = train_network(X, Y)
            save_sequential_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

            var = 'velocity'
            center = half
            X, Y, mins, maxs, my, My = dataset.get_sequential_dataset(stencil_size, center, var)
            model = train_network(X, Y)
            save_sequential_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

            var = 'velocity'
            center = stencil_size - half - 1
            X, Y, mins, maxs, my, My = dataset.get_sequential_dataset(stencil_size, center, var)
            model = train_network(X, Y)
            save_sequential_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

        else:
            var = 'pressure'
            center = half
            X, Y, mins, maxs, my, My = dataset.get_sequential_dataset(stencil_size, center, var)
            model = train_network(X, Y)
            save_sequential_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

            var = 'velocity'
            center = half
            X, Y, mins, maxs, my, My = dataset.get_sequential_dataset(stencil_size, center, var)
            model = train_network(X, Y)
            save_sequential_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)
