"""
Deep Q Learning for Atari Breakout
"""

import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from DQN import Create_DQN
import warnings
warnings.filterwarnings("ignore")


# Creating Environment
env = gym.make("Breakout-v4")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, 4)

# No.of Actions
NumActions = 4

# Importing Model
Model = Create_DQN()					# Used to make Prediction of Q-Values which are used to make action / Critic Model
Target_Model = Create_DQN()				# Model for prediction of future rewards / Actor Model

# Optimizer
Optimizer = tf.keras.optimizers.Adam()

# Loss Function
LossFunction = tf.keras.losses.Huber()

# Parameters
discount_rate = 0.99
epsilon = 1.0
min_epsilon = 0.1
max_epsilon = 1.0

# Training Parameters
Episodes = 250							# No.of Episodes
TimeSteps = 10000						# Each Episode has a maximum of 1000 TimeSteps
Batch_Size = 32							# Batch_Size from reply buffer

# Frame Parameters
frame_count = 0							# No.of Frames
random_frames = 50000					# No.of Frames to perform a random action to observe the output
greedy_frames = 1000000					# No.of Frames for Exploration
max_memory = 100000						# Maximum Memory to store
Steps_per_Model_Update = 4				# No.of Steps before Training the Model
Steps_per_TargetModel_Update = 10000	# No.of Steps before Updating the Model
Episodes_Rewards = []

# Reply Buffer
Replay = {}
History = ["state", "action", "next_state", "reward", "done"]
for p in History:
	Replay[p] = []


for episode in range(Episodes):

	state = np.array(env.reset())
	episode_reward = 0

	for t in range(TimeSteps):
		# env.render()

		frame_count +=1
		print ("Frame Count = {}".format(frame_count))

		if frame_count < random_frames:
			action = np.random.choice(NumActions)
		else:
			"""
			Epsilon-Greedy Algorithm
			"""
			if epsilon > np.random.rand(1)[0]:
				# Epsilon Step
				action = np.random.choice(NumActions)
			else:
				# Greedy Step => Perform Action with most Q-Value
				action_probabilities = Model(np.expand_dims(state,axis=0))
				action = tf.argmax(action_probabilities)
		
		# Epsilon-Decay with increase of no.of Frames => Decaying Probability of Random Action
		epsilon -= (max_epsilon - min_epsilon)/greedy_frames
		epsilon = max(epsilon, min_epsilon)

		# Applying Action to the Environment
		next_state, reward, done, info = env.step(action)
		next_state = np.array(next_state)

		# Increasing Episode Reward
		episode_reward += reward

		# Updating Data in Replay
		Replay["state"].append(state)
		Replay["action"].append(action)
		Replay["next_state"].append(next_state)
		Replay["reward"].append(reward)
		Replay["done"].append(done)

		# Size of Memory
		Memory_Size = len(Replay["state"])


		# Updating the Model
		if frame_count % Steps_per_Model_Update == 0 and Memory_Size > Batch_Size:

			# Samples to Train
			Indices = np.random.choice(Memory_Size, size=Batch_Size)

			State_Samples = np.array(Replay["state"])[Indices]
			Action_Samples = np.array(Replay["action"])[Indices]
			Next_State_Samples = np.array(Replay["next_state"])[Indices]
			Rewards_Samples = np.array(Replay["reward"])[Indices]
			Done_Samples = np.array(Replay["done"], dtype=np.float)[Indices]

			"""
			s		: State
			a		: Action
			s'		: Next_State
			r		: Reward
			gamma	: discount_rate
			"""

			# max_(a') Q(s',a',TargetModel)
			future_rewards = tf.reduce_max(Target_Model.predict(Next_State_Samples), axis=1)

			# y = r + gamma * max_(a') Q(s',a',TargetModel)
			y = Rewards_Samples + discount_rate * future_rewards

			# Q value of Last-Frame is set to "-1" => When it is last frame before done
			y = y * (1 - Done_Samples) - Done_Samples

			# Masks of Actions
			Masks = tf.one_hot(Action_Samples, NumActions)

			with tf.GradientTape as tape:
				# Q(s,TargetModel)
				Q_Values = Model(State_Samples)

				# Q(s,a,TargetModel)
				y_pred = tf.reduce_sum(tf.multiply(Q_Values, Masks), axis=1)

				# Loss
				loss = LossFunction(y, y_pred)
			
			# Updating Model
			grads = tape.gradient(loss, Model.trainable_variables)
			Optimizer.apply_gradients(zip(grads, Model.trainable_variables))

		
		# Updating Target Model
		if frame_count % Steps_per_TargetModel_Update == 0:
			Target_Model.set_weights(Model.get_weights())
			print ("Mean Episode Reward = {},	No.of Episodes = {},	No.of Frames = {}".format(np.mean(Episodes_Rewards), epoch, frame_count))


		# Checking Memory
		if Memory_Size > max_memory:
			del Replay["state"][:1]
			del Replay["action"][:1]
			del Replay["next_state"][:1]
			del Replay["reward"][:1]
			del Replay["done"][:1]

		# Done
		if done:
			break

	# Updating Episodes Rewards
	Episodes_Rewards.append(episode_reward)
	
	if len(Episodes_Rewards) > 100:
		del Episodes_Rewards[:1]
	
	if np.mean(Episodes_Rewards) > 40:
		print ("Solved at Episode-{}".format(episode))
		break


	print ("Episode-" + str(episode) + ":	Episode-Reward = " + str(episode_reward))