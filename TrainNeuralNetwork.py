import numpy as np

file_names = "200PokeData"

train_images = np.load("data/Images" + file_names + ".npy")
train_labels = np.load("data/Labels" + file_names + ".npy")

print(train_images.shape)
print(train_labels.shape)
print(train_labels[5])
