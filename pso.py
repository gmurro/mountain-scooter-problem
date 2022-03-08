from random import random
from random import uniform
import numpy as np

from mountain_car import MountainCar

class Particle:
    def __init__(self, id, size, position_bounds):
        self.id = id
        self.size = size
        self.position_bounds = position_bounds
        self.position = np.random.randint(position_bounds[0], position_bounds[1] + 1, size)
        self.velocity = np.random.uniform(-1, 1, size)
        self.best_position = self.position
        self.best_fitness = -np.inf
        self.fitness = None

    # evaluate current fitness
    def evaluate(self, fitness_function):
        self.fitness = fitness_function(self.position)

        # check to see if the current position is an individual best
        if self.fitness > self.best_fitness:
            self.best_position = self.position
            self.best_fitness = self.fitness

    # update new particle velocity
    def update_velocity(self, w, c1, c2, population_best):

        r1 = np.random.random(self.size)
        r2 = np.random.random(self.size)

        cognition_term = c1 * r1 * (self.best_position - self.position)
        social_term = c2 * r2 * (population_best - self.position)
        self.velocity = w * self.velocity + cognition_term + social_term

    # update the particle position based off new velocity updates
    def update_position(self):
        self.position = self.position + self.velocity

        # round to nearest integer and clip into bounds
        self.position = np.rint(self.position)
        self.position = np.clip(self.position, self.position_bounds[0], self.position_bounds[1])

    def __str__(self):
        return "Particle {} (fitness={})".format(self.id, self.fitness)

    def __repr__(self):
        return self.__str__()

class PSO:
    def __init__(self, size_particle, position_bounds, fitness_function, num_particles=100, max_iterations=100, w=0.5, c1=1.5, c2=1.5):
        self.size_particle = size_particle
        self.positions_bounds = position_bounds
        self.fitness_function = fitness_function
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = self.initialize()
        self.population_best = self.compute_population_best()

    def initialize(self):
        particles = []
        for i in range(self.num_particles):
            particles.append(Particle(i, self.size_particle, self.positions_bounds))
            particles[i].evaluate(self.fitness_function)
        return particles

    def compute_population_best(self):
        return self.particles[np.argmax([particle.fitness for particle in self.particles])]

    def run(self):
        for i in range(self.max_iterations):
            for particle in self.particles:
                particle.update_velocity(self.w, self.c1, self.c2, self.population_best.position)
                particle.update_position()
                particle.evaluate(self.fitness_function)
            self.population_best = self.compute_population_best()
            print("Iteration {}: {}".format(i, self.population_best))


def evaluate_policy(policy, env, num_bins):

    # list of thresholds according to which packing in bins the velocity and the position
    velocity_state_array = np.linspace(env.max_speed, env.max_speed, num=num_bins - 1, endpoint=False)
    position_state_array = np.linspace(env.min_position, env.max_position, num=num_bins - 1, endpoint=False)

    # Reset and return the first observation
    velocity, position = env.reset(exploring_starts=True)

    # The observation is digitized, meaning that an integer corresponding
    # to the bin where the raw float belongs is obtained and use as replacement.
    state = (np.digitize(velocity, velocity_state_array), np.digitize(position, position_state_array))

    max_steps = 200
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

def main():
    # initialize environment
    env = MountainCar(mass=0.70, friction=0.35, max_speed=2.8)

    num_bins = 20
    num_particles = 100

    # initialize PSO
    pso = PSO(
        size_particle=(num_bins, num_bins)
        , position_bounds=(0, 2)
        , fitness_function=lambda policy: evaluate_policy(policy, env, num_bins)
        , num_particles=num_particles
        , max_iterations=100
        , w=0.73
        , c1=1.5
        , c2=1.5
    )
    pso.run()

    evaluate_policy(pso.population_best.position, env, num_bins)
    print(pso.population_best)
    env.render(file_path='./mountain_car.gif', mode='gif')

if __name__ == "__main__":
    main()