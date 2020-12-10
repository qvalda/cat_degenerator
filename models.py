from decorator import init
from keras import Sequential, Input, Model, layers
from keras.optimizers import Adam
from keras.layers import Conv2D, LeakyReLU, Dropout, Conv2DTranspose, BatchNormalization, Flatten, Dense, Activation, \
    Reshape, \
    UpSampling2D

img_rows = 64
img_cols = 64
channels = 3
z_dimension = 64
epochs_samples = "epochs_samples/"
generated_samples = "generated_samples/"


def build_discriminator():
    img_shape = (img_rows, img_cols, channels)
    return Sequential([

        layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape),
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
    # return Model(input=img, output=d_pred)


def build_generator():
    return Sequential([
        layers.Dense(8*8*512, input_shape=(z_dimension,)),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((8, 8, 512)),

        layers.Conv2DTranspose(256, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(128, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(64, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(32, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(channels, (5, 5), padding='same', activation='tanh'),

    ], name="generator")

def build_combined(generator, discriminator):
    # the generator takes noise as input and generates imgs
    z = Input(shape=(z_dimension,))
    img = generator(z)
    discriminator.trainable = False
    d_pred = discriminator(img)
    # The combined model  (stacked generator and discriminator)
    return Model(z, d_pred)


def create_network():
    optimizer = Adam(0.0005, 0.5)

    # build discriminator
    discriminator = build_discriminator()
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    # bild generator
    generator = build_generator()
    generator.summary()

    # The combined model  (stacked generator and discriminator)
    combined = build_combined(generator, discriminator)
    combined.compile(loss='binary_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])

    return generator, discriminator, combined
