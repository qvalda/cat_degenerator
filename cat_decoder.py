from PIL import Image, ImageFilter
import numpy as np

#Read image
im = Image.open( 'cats/1.jpg' )
#Display image
#im.show()

im2arr = np.array(im) # im2arr.shape: height x width x channel
arr2im = Image.fromarray(im2arr)

#Applying a filter to the image
#im_sharp = im.filter( ImageFilter.SHARPEN )
#Saving the filtered image to a new file
#im_sharp.save( 'image_sharpened.jpg', 'JPEG' )

#Splitting the image into its respective bands, i.e. Red, Green,
#and Blue for RGB
r,g,b = im.split()

#Viewing EXIF data embedded in image
exif_data = im._getexif()
exif_data
#def cat_decoder():
