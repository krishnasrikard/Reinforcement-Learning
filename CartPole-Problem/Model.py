"""
Model for Cart-Pole Pole.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense

def Create_Model():
	tf.keras.backend.clear_session()
	In = tf.keras.Input(shape=(4), name="input_layer")
	x = Dense(4,activation="elu", name="hidden_layer")(In)
	Out = Dense(1,activation='sigmoid', name="output_layer")(x)

	return tf.keras.models.Model(inputs=In, outputs=Out)