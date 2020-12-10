import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import Sequential, Input, Model, layers
from keras.optimizers import Adam

img_rows = 64
img_cols = 64
channels = 3
latent_dim = 64
epochs_samples = "epochs_samples/"
generated_samples = "generated_samples/"
image_shape = (img_rows, img_cols, channels)


# image processing block

# transform image to array with scaling

def decode_cats():
    cats_files = [f for f in os.listdir("cats") if f.endswith('.jpg')]
    cats_count = len(cats_files)
    cats_inputs = np.zeros((cats_count, img_rows, img_cols, channels))
    count = 0

    for cats_file in cats_files:
        im = Image.open('cats/' + cats_file)
        im2arr = np.array(im)
        im2arr = im2arr / 127.5 - 1
        cats_inputs[count] = im2arr
        count = count + 1

    return cats_inputs


# creating the table of current generated cats
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


# saving an image from array
def encode_cat(array, filename):
    array = (array + 1) * 127.5
    array = array.astype(np.ubyte)
    image = Image.fromarray(array)
    image.save(filename, 'JPEG')


# presenting generator progress
def sample_images(epoch):
    encode_cats(lambda: generator.predict(np.random.normal(0, 1, (1, latent_dim))), epochs_samples + "%d.jpg" % epoch)


# eob image processing_____________________________________


# models configuration block

def discriminator_model():
    return Sequential([
        layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=image_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),

        layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),

        layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(1),
        layers.Activation("sigmoid"),
    ], name="discriminator")


def generator_model():
    return Sequential([
        layers.Dense(8 * 8 * 512, input_shape=(latent_dim,)),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((8, 8, 512)),

        # layers.UpSampling2D(),
        layers.Conv2DTranspose(256, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # layers.UpSampling2D(),
        layers.Conv2DTranspose(128, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # layers.UpSampling2D(),
        layers.Conv2DTranspose(64, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(32, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(channels, (5, 5), padding='same', activation='tanh'),

    ], name="generator")


# The training_generator model  (stacked generator and discriminator)

def training_generator_model(generator, discriminator):
    latent_space = Input(shape=(latent_dim,))
    img = generator(latent_space)
    # froze discriminator's state for training simplification
    discriminator.trainable = False
    d_pred = discriminator(img)
    return Model(latent_space, d_pred)

# eob models configuration_________________________________


def create_networks():
    optimizer = Adam(0.0005, 0.5)

    # build discriminator
    discriminator = discriminator_model()
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    # build generator
    generator = generator_model()
    generator.summary()

    # build training_generator model
    training_generator = training_generator_model(generator, discriminator)
    training_generator.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

    return generator, discriminator, training_generator


def learn():
    cats_train_set = decode_cats()
    cats_count = cats_train_set.shape[0]

    # expected discriminator answers
    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # training
    for epoch in range(epochs + 1):
        # train discriminator on real images
        idx = np.random.randint(0, cats_count, batch_size)
        imgs = cats_train_set[idx]
        discr_loss_real = discriminator.train_on_batch(imgs, true)

        # train discriminator on fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        discr_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        # train generator with frozen discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_loss = training_generator.train_on_batch(noise, true)

        # log loss functions values and show generated results for period
        if (epoch % logging_period) == 0:
            discr_loss = discr_loss_real + discr_loss_fake
            print("%Epoch d : D loss = %f, G loss = %f" % (epoch, discr_loss[0], gen_loss[0]))
            sample_images(epoch)

    generator.save("generator_%d_epoch.h5" % epochs)
    discriminator.save("discriminator_%d_epoch.h5" % epochs)
    training_generator.save("training_generator_%d_epoch.h5" % epochs)


def load():
    epoch = 30000
    generator.load_weights("generator_%d_epoch.h5" % epoch)

    for number in range(10000):
        cat = generator.predict(np.random.normal(0, 1, (1, latent_dim))).reshape(64, 64, -1)
        encode_cat(cat, generated_samples + "/%d.jpg" % number)


# main

generator, discriminator, training_generator = create_networks()

# learning parameters
epochs = 30000
batch_size = 100
logging_period = 100

# call of learning process
learn()

# load cats from trained generator
load()
