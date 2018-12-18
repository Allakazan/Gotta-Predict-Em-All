import numpy as np
import pandas as pd
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os

from progress import progress


pokemon_dir = 'pokemon'

def convert_to_jpg(img_path):
    # Convert png to jpeg
    img = Image.open(img_path)

    if img.mode == 'RGBA':
        img.load()
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = np.array(background)
    elif img.mode == 'P':
        img = img.convert('RGBA')
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = np.array(background)
    else:
        img = img.convert('RGB')
        img = np.array(img)

    return img


# Resize image to 128x128
def resize_img(img):
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


# Normalize pixel values from 1 to 0, important when utilizing NNs
def normalize_img(img):
    img = img / 255.0

    return img


# Open an image, convert to jpeg, resize if needed
def open_convert(img_path):
    # png
    if img_path[-4:].lower() == '.png' or img_path[-4:].lower() == '.gif':
        img = convert_to_jpg(img_path)
    # jpeg, etc.
    else:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)

    # Convert to 128x128
    img = resize_img(img)

    # Normalize img
    img = normalize_img(img)

    # Return resized img
    return img


# Contain images and labels
images = []
labels = []

# How many images per pokemon to load
images_per_pokemon = 15

# Keep track of current iteration
count = 0
# Iterate through each pokemon folder
for pkmn in os.listdir(pokemon_dir):
    pkmn_dir = os.path.join(pokemon_dir, pkmn)

    # Current number of images loaded for this pokemon
    curr_imgs = 0

    progress.printProgressBar(0, images_per_pokemon, prefix="{:<15}".format(pkmn+':'), suffix='Complete', length=50)

    # Add each image to the list, use most relevant search results
    for img in sorted(os.listdir(pkmn_dir)):
        # Attempt to add image and label to list
        try:
            images.append(open_convert(os.path.join(pkmn_dir, img)))
            labels.append(pkmn)
        # Ignore garbage images
        except (ValueError, OSError):
            continue
        count += 1

        # Some visualization for time spent loading
        progress.printProgressBar(curr_imgs + 1, images_per_pokemon, prefix="{:<15}".format(pkmn+':'), suffix='Complete', length=50)

        # Increment num images loaded
        curr_imgs += 1
        if curr_imgs >= images_per_pokemon:
            break


num_rows = 3
num_cols = 10
num_images = num_rows*num_cols
plt.figure(figsize=(1.7*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, num_cols, i+1)
  plt.grid(False)
  plt.imshow(images[i], cmap='gray')
  plt.xlabel(labels[i])
  plt.xticks([])
  plt.yticks([])

plt.show()
