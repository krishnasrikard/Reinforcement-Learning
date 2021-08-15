"""
Understanding Breakout Environment
- Observations are screenshots of scene.
- 4 possible actions are avaiable.
"""

import gym
import warnings
warnings.filterwarnings("ignore")

# Creating Environment
env = gym.make("Breakout-v4")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, 4)

# Environment Spaces
print ("Action Spaces of Environment:", env.action_space)
print ("Observation Spaces of Environment:", env.observation_space)
print ("Observation Space Maximum:", env.observation_space.high)
print ("Observation Space Minimum:", env.observation_space.low)
print ("Observation Space Shape:", env.observation_space.low.shape)
print ("-"*50)

