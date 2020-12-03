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