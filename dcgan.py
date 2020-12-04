# imports
import tensorflow as tf
from tf import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape

# we're going to implement a DCGAN class 
class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, inputDim=100, outputDim=512):
        # initalizing the model
        model = Sequential()
        inputshape = (dim, dim, depth)
        chan_dim = -1
        # first layer
        model.add(Dense(input_dim=input_dim, units=outputDim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        # second layer
        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        # rehsaping layers and upsampling
        model.add(Reshape(inputshape))
        model.add(Conv2dTranspose(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        # rehsaping and upsampling but with tanh activation 
        model.add(Conv2dTranspose(channels, (5,5), strides=(2,2), padding="same"))
        model.add(Activation("tanh"))

        return model


   @staticmethod
   def build_discriminator(width, height, depth, alpha=0.2):
       # initalizing the model
       model = Sequential()
       inputshape = (height, width, depth)

       # first layer
       model.add(Conv2D(32, (5, 5), padding="same", strides=(2,2), input_shape=inputshape))
       model.add(LeakyReLU(alpha=alpha))

       #second layer
       model.add(Conv2D(64, (5, 5), padding="same", strides=(2,2)))     
       model.add(LeakyReLU(alpha=alpha))

       # third layer
       model.add(Flatten())
       model.add(Dense(512))
       model.add(LeakyReLU(alpha=alpha))

       # output layer
       model.add(Dense(1))
       model.add(Activation("sigmoid"))

       return model
