# imports
import DCGAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from sklearn.utils import shuffle
from imutils import build_montages

import numpy as np
import argparse
import cv2
import os

# arguments for parsing the command line
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=50, help="# of epochs to train")
ap.add_argument("-b", "--batch-size", type=int, default=128, help="batch size for training")

args = vars(ap.parse_args())

NUM_EPOCHS = args['epochs']
BATCH_SIZE = args['batch-size']
LR_INIT = 2e-4

# load fashion mnist dataset and stack training and testing datapoints so we have additonal training
print("loading mnist dataset...")
((X_train, _), (X_test, _)) = fashion_mnist.load_data()
train_images = np.concatenate([X_train, X_test])

# add extra dimension for channel and scale the images
train_images = np.expand_dims(train_images, axis=-1)
train_images = (train_images.astype("float") - 127.5) / 127.5

# build the generator
print("building generator...")
generator = DCGAN.build_generator(7, 64, channels=1)

# build the discriminator
print("building discriminator...")
discriminator = DCGAN.build_discriminator(28, 28, 1)
optimizer = Adam(lr = LR_INIT , beta=0.5, decay= LR_INIT / NUM_EPOCHS)
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

# building the GAN
print("building gan...")
discriminator.trainable=False 
gan_input = Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

# compile gan
gan_optimizer = Adam(lr = LR_INIT, beta=0.5, decay = LR_INIT / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=gan_optimizer)

# generating noise so we can see how our generator is performing
print("training...")
justnoise = np.random.uniform(-1, 1, size=(256, 100))

# loop it over the epochs
for epoch in range(0, NUM_EPOCHS):
    # computing number of batches per epoch
    print("starting epoch {} of {}...".format(epoch + 1, NUM_EPOCHS))
    batches_x_epoch = int(train_images.shape[0] / BATCH_SIZE)

    # loop over batches
    for i in range(0, batches_x_epoch):
        # initialize empty output path
        p = None

        # select next batch of images, and randomly generate noise for generator to predixt on
        image_batch = train_images[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
        
        # generate images using noise + generator model
        generator_images = generator.predict(noise, verbose=0)

        # concatenate the actual images and the generated images then shuffle data
        X = np.concatenate((image_batch, generator_images))
        y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
        y = np.reshape(y, (-1,))
        (X, y) = shuffle(X, y)

        # train discriminator on the data
        discriminator_loss = discriminator.train_on_batch(X, y)

        # training our generator via the adversarial model 
        sudo_labels = [1] * BATCH_SIZE
        sudo_labels = np.reshape(sudo_labels, (-1,))
        gan_loss = gan.train_on_batch(noise, sudo_labels)

        # checking output to see if end of epoch, if so intialize output path
        if i == batches_x_epoch - 1:
            p = [args["output"], "epoch_{}_output.png".format(str(epoch + 1).zfill(4))]

        # otherwise check to visualize the current batch of the epoch
        else:
            if epoch < 10 and i % 25 == 0:
                p = [args ["output"], "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]

            elif epoch >= 10 and i % 100 == 0:
                p = [args ["output"], "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]

    
        
        

