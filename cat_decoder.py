import os

from PIL import Image
import numpy as np

cats_count = 15747


def decode_cats():
    cats_files = os.listdir('cats')
    cats_inputs = np.zeros((cats_count,64,64,3))
    count = 0

    for cats_file in cats_files:
        im = Image.open('cats/' + cats_file)
        im2arr = np.array(im)  # im2arr.shape: height x width x channel
        im2arr = im2arr / 127.5 - 1
        # im2arr = im2arr.reshape-1, 64, 64)
        cats_inputs[count]=im2arr
        count=count+1
        # if count>=200:
        #     break

    return cats_inputs

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
