from cat_decoder import *
from models import *

generator, discriminator, combined = create_network()


def sample_images(epoch):
    encode_cats(lambda: generator.predict(np.random.normal(0, 1, (1, z_dimension))), samples_directory + "%d.png" % epoch)


def learn():
    cats_train_set = decode_cats()
    cats_count = cats_train_set.shape[0]
    # training parameters
    epochs = 40000
    batch_size = 128
    sample_interval = 100  # save some generated pictures

    # adversarial ground truths
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # training
    for epoch in range(epochs + 1):
        # real images
        idx = np.random.randint(0, cats_count, batch_size)
        imgs = cats_train_set[idx]
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

    generator.save("generator_%d_epoch.h5" % epochs)
    discriminator.save("discriminator_%d_epoch.h5" % epochs)
    combined.save("combined_%d_epoch.h5" % epochs)


def load():
    epoch = 40000
    generator.load_weights("generator_%d_epoch.h5" % epoch)

    for number in range(10000):
        cat = generator.predict(np.random.normal(0, 1, (1, z_dimension))).reshape(64, 64, -1)
        encode_cat(cat, samples_directory + "/generated/%d.jpg" % number)


learn()
#load()
