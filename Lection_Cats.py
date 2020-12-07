# Imports
from keras.layers import Activation, Dense, Conv2D, UpSampling2D, LeakyReLU, Reshape, Flatten, Input, \
    BatchNormalization, Dropout, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import numpy as np


# msvcp_dll_name = 'msvcp140.dll'
# cudart_dll_name = 'cudart64_90.dll'
# cuda_version_number = '9.0'
# nvcuda_dll_name = 'nvcuda.dll'
# cudnn_dll_name = 'cudnn64_8.dll'
# cudnn_version_number = '8'
from cat_decoder import decode_cats
from cat_decoder import cats_count

def build_discriminator(img_shape):
    model = Sequential()  # 64x64 original shape

    model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())#momentum=0.8
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())#momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())#momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))


    model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())#momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1))
    #model.add(Activation("sigmoid"))

    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(inputs=img, outputs=d_pred)


def build_generator(z_dimension, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 1024, input_dim=z_dimension))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 1024)))

    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())#momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    #model.add(UpSampling2D())
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())#momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())#momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))


    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same',activation='tanh'))

    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(inputs=noise, outputs=img)


def sample_images(epoch):
    r, c = 4, 5
    noise = np.random.normal(0, 1, (r * c, z_dimension))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(PATH + "%d.png" % epoch, dpi=200)
    plt.close()


# load real pictures:

x_train = decode_cats()

# model parameters
PATH = "D:/Temp/cats/"
img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)
z_dimension = 64
optimizer = Adam(0.0005, 0.5)

# build discriminator
discriminator = build_discriminator(img_shape)
discriminator.summary()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

# bild generator
generator = build_generator(z_dimension, channels)
generator.summary()

# the generator takes noise as input and generates imgs
z = Input(shape=(z_dimension,))
img = generator(z)
discriminator.trainable = False
d_pred = discriminator(img)
# The combined model  (stacked generator and discriminator)
combined = Model(z, d_pred)
combined.compile(loss='binary_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])

# training parameters
epochs = 30000
batch_size = 64
sample_interval = 100  # save some generated pictrures

# adversarial ground truths
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# training
for epoch in range(epochs):
    # real images
    idx = np.random.randint(0, cats_count, batch_size)
    imgs = x_train[idx]
    # generated images
    noise = np.random.normal(0, 1, (batch_size, z_dimension))
    gen_imgs = generator.predict(noise)
    # train discriminator
    d_loss_real = discriminator.train_on_batch(imgs, real)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # train generator
    noise = np.random.normal(0, 1, (batch_size, z_dimension))
    g_loss = combined.train_on_batch(noise, real)
    # save progress
    if (epoch % sample_interval) == 0:
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
        sample_images(epoch)
        # generator.save("generator_64_64_z64_%d_epoch.h5" % epoch)
