"""
Understanding CartPole Environment
"""

import gym
import warnings
warnings.filterwarnings("ignore")

# Creating Environment
env = gym.make('CartPole-v0')

# Environment Spaces
print ("Action Spaces of Environment:", env.action_space)
print ("Observation Spaces of Environment:", env.observation_space)
print ("Observation Space Maximum:", env.observation_space.high)
print ("Observation Space Minimum:", env.observation_space.low)
print ("-"*50)

# Observations from Environment
obs = env.reset()
print ("Parameters:",obs)
print ("Position:",obs[0])
print ("Velocity:",obs[1])
print ("Angle:",obs[2])
print ("Angular Velocity:",obs[3])
print ("-"*50)

# Creating a Random Simulation
Episodes = 100
TimeSteps = 1000
for e in range(Episodes):
	env.reset()
	for t in range(TimeSteps):
		env.render()
		
		# Selects a Random Action
		action = env.action_space.sample()
		
		observation, reward, done, info = env.step(action)
		
		if done:
			"""
			- When done is True it indicates that episode is terminated. (Pole might have tipped to far..)
			"""
			print ("Episode-{} finished after {} timesteps".format(e+1,t+1))
			break
env.close()
