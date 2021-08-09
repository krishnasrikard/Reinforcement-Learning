"""
Basic Policy for CartPole Environment
- If pole is tilted right we move right else we move left.
"""

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Creating Environment
env = gym.make('CartPole-v0')


Episodes = 100
TimeSteps = 1000
TotalRewards = np.zeros((Episodes,))

for e in range(Episodes):
	observation = env.reset()
	for t in range(TimeSteps):
		env.render()
		
		if observation[2] < 0:
			# Move right if tilted left
			action = 0
		else:
			# Move left if tilted right
			action = 1
		
		observation, reward, done, info = env.step(action)
		TotalRewards[e] += reward
		
		if done:
			"""
			- When done is True it indicates that episode is terminated. (Pole might have tipped to far..)
			"""
			print ("Episode-{} finished after {} timesteps".format(e+1,t+1))
			break
env.close()

print ("Mean Reward per Episode =", np.mean(TotalRewards))
print ("Standard Reward per Episode =", np.std(TotalRewards))
print ("Max Reward per Episode =", np.max(TotalRewards))
print ("Min Reward per Episode =", np.min(TotalRewards))
