"""
Basic Policy-Gradients for CartPole Environment
- We let NN play multiple times and save gradients at each step.
- We estimate action-score. We multiply gradients with action-score and update model with their mean.
"""

import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from Model import Create_Model
from Discounts import *
import warnings
warnings.filterwarnings("ignore")


# Creating Environment
env = gym.make('CartPole-v0')

# Importing Model
Model = Create_Model()

# Optimizer
Optimizer = tf.keras.optimizers.Adam()

# Discount_Rate
discount_rate = 0.95

# Training Parameters
Epochs = 250			# Model is updated after each Epoch
Episodes = 10			# Each Epochs has 10 Episodes
TimeSteps = 1000		# Each Episode has a maximum of 1000 TimeSteps

for epoch in range(Epochs):
	TotalRewards = 0
	epoch_rewards = []
	epoch_gradients = []

	for episode in range(Episodes):
		episode_rewards = []
		episode_gradients = []
		obs = env.reset()
		
		for t in range(TimeSteps):
			env.render()
			with tf.GradientTape() as tape:
				obs = np.expand_dims(obs,axis=0)
				y_pred = Model(obs, training=True)

				# We are assuming that chosen action is best possible.
				p = y_pred.numpy()[0][0]
				action = tf.random.categorical(tf.math.log([[p,1-p]]), 1).numpy()[0][0]
				y = tf.constant([[1.0 - action]])

				# Loss
				loss = tf.keras.losses.binary_crossentropy(y,y_pred)

			# Computing Gradients
			grads = tape.gradient(loss, Model.trainable_weights)

			# Applying it to Environment
			obs, reward, done, info = env.step(action)
			TotalRewards += reward

			episode_rewards.append(reward)
			episode_gradients.append(grads)
			if done:
				break

		epoch_rewards.append(episode_rewards)
		epoch_gradients.append(episode_gradients)

	# Updating Gradients on Reward Basis
	epoch_rewards = normalize_all_discount_rewards(epoch_rewards, discount_rate)
	
	Updated_Gradients = []
	for episode in range(len(epoch_rewards)):
		for t in range(len(epoch_rewards[episode])):
			Grad = []
			for g in epoch_gradients[episode][t]:
				Grad.append(tf.multiply(epoch_rewards[episode][t], g))
			Updated_Gradients.append(Grad)
	
	Updated_Gradients = np.mean(Updated_Gradients,axis=0)
	Optimizer.apply_gradients(zip(Updated_Gradients, Model.trainable_variables))
	print ("Epochs-"+str(epoch)+":	" + str(TotalRewards))

Model.save("Model")
