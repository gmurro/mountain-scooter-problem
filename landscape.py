import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
from tqdm import tqdm

matplotlib.use('Qt5Agg')

from mountain_car import MountainCar

# initialize environment
env = MountainCar(mass=0.45, friction=0.4, max_speed=1.8)

def fitness(policy):

    num_bins = policy.shape[0]

    # list of thresholds according to which packing in bins the velocity and the position
    velocity_state_array = np.linspace(env.max_speed, env.max_speed, num=num_bins - 1, endpoint=False)
    position_state_array = np.linspace(env.min_position, env.max_position, num=num_bins - 1, endpoint=False)

    # Reset and return the first observation
    velocity, position = env.reset(exploring_starts=False)

    # The observation is digitized, meaning that an integer corresponding
    # to the bin where the raw float belongs is obtained and use as replacement.
    state = (np.digitize(velocity, velocity_state_array), np.digitize(position, position_state_array))

    max_steps = 100
    cumulated_reward = 0
    for step in range(max_steps):

        action = int(policy[state])
        # Move one step in the environment and get the new state and reward
        (new_velocity, new_position), reward, done = env.step(action)
        state = (np.digitize(new_velocity, velocity_state_array),
                 np.digitize(new_position, position_state_array))
        cumulated_reward += reward

        # if the episode is done, break the loop
        if done: break
    return cumulated_reward


num_bins = 2

actions = [0, 2]
combinatorial_policy = product(actions, repeat=num_bins*num_bins)
policies = [np.fromiter(policy, np.int8).reshape(num_bins, num_bins) for policy in tqdm(combinatorial_policy, total=num_bins*num_bins)]

x = np.arange(0, len(policies))
y = [fitness(p) for p in tqdm(policies)]

plt.plot(x, y)
plt.show()
"""
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)

X, Y = np.meshgrid(x, y)
Z = akley_function(X, Y)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
"""