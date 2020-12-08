import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def decode_cats():
    cats_files = os.listdir('cats')
    cats_count = len(cats_files)
    cats_inputs = np.zeros((cats_count, 64, 64, 3))
    count = 0

    for cats_file in cats_files:
        im = Image.open('cats/' + cats_file)
        im2arr = np.array(im)  # im2arr.shape: height x width x channel
        im2arr = im2arr / 127.5 - 1
        # im2arr = im2arr.reshape-1, 64, 64)
        cats_inputs[count] = im2arr
        count = count + 1
        # if count>=200:
        #     break

    return cats_inputs


def encode_cats(r, c, generator, filename):
    # array = array.reshape(64, 64, -1)
    # array = (array + 1) * 127.5
    # array = array.astype(np.ubyte)
    # arr2im = Image.fromarray(array)
    # arr2im.save( 'image_sharpened.jpg', 'JPEG' )

    fig, axs = plt.subplots(r, c)
    # fig.tight_layout()
    # plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=None, hspace=None)
    for i in range(r):
        for j in range(c):
            array = generator(i, j)
            # array = array.reshape(64, 64, -1)
            array = array[0]
            array = (array + 1) * 127.5
            array = array.astype(np.ubyte)
            # arr2im = Image.fromarray(array)
            axs[i, j].imshow(array)
            axs[i, j].axis('off')
    # fig.tight_layout()
    # plt.show()
    fig.savefig(filename, dpi=200)

    plt.close()


def encode_cat(array, filename):
    array = (array + 1) * 127.5
    array = array.astype(np.ubyte)
    image = Image.fromarray(array)
    image.save(filename, 'JPEG')

# Read image
# im = Image.open('cats/1.jpg')
# Display image
# im.show()

# im2arr = np.array(im)  # im2arr.shape: height x width x channel
# im2arr = im2arr / 127.5 - 1
#
# arr2im = Image.fromarray(im2arr)

# Applying a filter to the image
# im_sharp = im.filter( ImageFilter.SHARPEN )
# Saving the filtered image to a new file
# im_sharp.save( 'image_sharpened.jpg', 'JPEG' )

# Splitting the image into its respective bands, i.e. Red, Green,
# and Blue for RGB
# r, g, b = im.split()

# Viewing EXIF data embedded in image
# exif_data = im._getexif()
# exif_data
# def cat_decoder():
