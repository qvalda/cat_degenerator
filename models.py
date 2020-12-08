from keras import Sequential, Input, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, LeakyReLU, Dropout, BatchNormalization, Flatten, Dense, Activation, Reshape, \
    UpSampling2D

img_rows = 64
img_cols = 64
channels = 3
z_dimension = 64
epochs_samples = "epochs_samples/"
generated_samples = "generated_samples/"


def build_discriminator():
    img_shape = (img_rows, img_cols, channels)

    model = Sequential()  # 64x64 original shape

    model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))  # 32x32
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))  # 16x16
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))  # 8x8
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(input=img, output=d_pred)


def build_generator():
    model = Sequential()

    model.add(Dense(128 * 16 * 16, input_dim=z_dimension))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 16, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=5, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(channels, kernel_size=5, padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(input=noise, output=img)


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
