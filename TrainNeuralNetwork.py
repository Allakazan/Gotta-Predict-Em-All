import numpy as np

filenames = "200PokeData"

train_images = np.load("data/Images" + filenames + ".npy")
train_labels = np.load("data/Labels" + filenames + ".npy")
