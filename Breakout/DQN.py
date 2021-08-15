"""
Creating a Deep Q-Network
- It takes State-Action pair as Input and gives Q(s,a) as Output.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Permute

def Create_DQN():
	tf.keras.backend.clear_session()
	In = tf.keras.Input(shape=(4,84,84))
	x = Permute((2,3,1))(In)
	x = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation="relu", padding='same')(x)
	x = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation="relu", padding='same')(x)
	x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation="relu", padding='same')(x)
	x = Flatten()(x)
	x = Dense(512, activation="relu")(x)
	Out = Dense(4, activation="linear")(x)

	return tf.keras.models.Model(inputs=In, outputs=Out)