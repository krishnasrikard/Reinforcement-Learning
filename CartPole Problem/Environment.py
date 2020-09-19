import gym
import numpy as np

#	Creating Environment
env = gym.make('CartPole-v0')
obs = env.reset()
print ("Parameters:",obs)
print ("Position:",obs[0])
print ("Velocity:",obs[1])
print ("Angle:",obs[2])
print ("Angular Velocity:",obs[3])
print ("--------------------------------------------------------------------------")
print ("Possible Actions",env.action_space)
print ("--------------------------------------------------------------------------")

#	Creating a Random Simulation
for _ in range(100):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    

# Creating a Ordered Simulation   
def basic_policy(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1

totals = []
for episode in range(5000):
	episode_rewards = 0
	obs = env.reset()
	for step in range(10000): # 1000 steps max, we don't want to run forever
		action = basic_policy(obs)
		obs, reward, done, info = env.step(action)
		episode_rewards += reward
		if done:
			break
	totals.append(episode_rewards)

print ("--------------------------------------------------------------------------")	
print ("Mean of Rewards of Ordered Simulation:",np.mean(totals))
print ("Standard Deviation of Rewards of Ordered Simulation:",np.std(totals))
print ("Max of Rewards of Ordered Simulation:",np.max(totals))
print ("Min of Rewards of Ordered Simulation:",np.min(totals))
print ("--------------------------------------------------------------------------")

env.close()
