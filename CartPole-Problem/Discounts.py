"""
Estimating Discounted Rewards and Normalised All Discounted Rewards
"""
import numpy as np

def discount_rewards(rewards, discount_rate):
	rewards = np.array(rewards)
	T = rewards.shape[0]
	discounted_rewards = np.zeros((T,))

	for t in reversed(range(T)):
		if t == T-1:
			discounted_rewards[t] = rewards[t]
		else:
			discounted_rewards[t] = rewards[t] + discount_rate * discounted_rewards[t+1]
		
	return discounted_rewards


def normalize_all_discount_rewards(all_rewards, discount_rate):
	all_discounted_rewards = np.array([discount_rewards(rewards, discount_rate) for rewards in all_rewards])
	
	mu = np.mean(np.concatenate(all_discounted_rewards))
	std = np.std(np.concatenate(all_discounted_rewards))

	return np.array([(discounted_rewards-mu)/std for discounted_rewards in all_discounted_rewards])