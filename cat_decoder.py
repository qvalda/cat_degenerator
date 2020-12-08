import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def decode_cats():
    cats_files = [f for f in os.listdir("cats") if f.endswith('.jpg')]
    cats_count = len(cats_files)
    cats_inputs = np.zeros((cats_count, 64, 64, 3))
    count = 0

    for cats_file in cats_files:
        im = Image.open('cats/' + cats_file)
        im2arr = np.array(im)
        im2arr = im2arr / 127.5 - 1
        cats_inputs[count] = im2arr
        count = count + 1

    return cats_inputs


def get_fake_cats():
    return get_cats("generated_samples")


def get_real_cats():
    return get_cats("cats")


def get_cats(name):
    cats_files = [f for f in os.listdir(name) if f.endswith('.jpg')]
    cats_count = len(cats_files)
    cats_inputs = np.zeros((cats_count, 64, 64, 3))
    count = 0

    for cats_file in cats_files:
        im = Image.open(name+'/' + cats_file)
        cats_inputs[count] = im
        count = count + 1

    return cats_inputs


def encode_cats(generator, filename):
    height = 4
    width = 5
    fig, axs = plt.subplots(height, width)
    for i in range(height):
        for j in range(width):
            array = generator()
            array = array[0]
            array = (array + 1) * 127.5
            array = array.astype(np.ubyte)
            axs[i, j].imshow(array)
            axs[i, j].axis('off')
    fig.savefig(filename, dpi=200)
    plt.close()


def encode_cat(array, filename):
    array = (array + 1) * 127.5
    array = array.astype(np.ubyte)
    image = Image.fromarray(array)
    image.save(filename, 'JPEG')
