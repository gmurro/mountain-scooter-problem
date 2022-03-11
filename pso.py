from random import random
from random import uniform
import numpy as np

from mountain_scooter import MountainScooter
from Particle import Particle

class PSO:
    """
    Class for the Particle Swarm Optimization algorithm.
    https://en.wikipedia.org/wiki/Particle_swarm_optimization
    """
    def __init__(self, n, num_particles, fitness_function, value_bounds=(0, 2),  max_iterations=50, w=0.7, c1=2.0, c2=2.0, verbose=False):
        """
        Initialize the Particle Swarm Optimization algorithm.
            :param n: Number of dimensions. It represent the size of a single particle.
            :param num_particles: Number of particles in the swarm.
            :param fitness_function: Fitness function to evaluate the particles.
            :param value_bounds: Tuple with the minimum and maximum values for each dimension. Default value is (0, 2).
            :param max_iterations: Maximum number of iterations. Default: 50.
            :param w: Inertia weight. Default: 0.7.
            :param c1: Cognitive weight. Default: 2.0.
            :param c2: Social weight. Default: 2.0.
            :param verbose: If True, print the progress of the algorithm. Default value is False.
        """
        self.n = n
        self.num_particles = num_particles
        self.fitness_function = fitness_function
        self.value_bounds = value_bounds
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        self.population = self.initialize_population()
        self.best_particle_population = self.compute_best_particle_population()

    def initialize_population(self):
        """
        Initialize the population of particles
            :return: A list of particles representing the population
        """
        population = []
        for i in range(self.num_particles):
            random_value = np.random.randint(low=self.value_bounds[0], high=self.value_bounds[1] + 1, size=self.n)
            population.append(Particle(i, random_value))
            population[i].evaluate(self.fitness_function)
        return population

    def compute_best_particle_population(self):
        """
        Compute the best particle in the population according to the fitness value.
            :return: Particle object representing the best particle in the population.
        """
        return self.population[np.argmax([particle.fitness for particle in self.population])]

    def optimize(self):
        for i in range(self.max_iterations):

            # print statistics
            if self.verbose:
                print(
                    f"🚀 Performing iteration {i+1}:\n\t📊 "
                    f"Avg={round(np.average([p.fitness for p in self.population]), 2)}\t"
                    f"Best value={self.best_particle_population.fitness}")

            for particle in self.population:
                # Update the particle velocity and according to that update the particle value
                particle.update_velocity(self.w, self.c1, self.c2, self.best_particle_population.value)
                particle.update_value()

                # Repair the particle if it is infeasible and evaluate its fitness
                particle.evaluate(self.fitness_function)
            self.best_particle_population = self.compute_best_particle_population()


def main():
    # initialize environment
    #env = MountainScooter(mass=0.4, friction=0.3, max_speed=1.8)
    env = MountainScooter(mass=0.5, friction=0.3, max_speed=2.5)

    # The biases have to be the same amount of the nodes without considering the first layer
    # The weights are the connections between the nodes of input and hidden layer + hidden and output layer
    n_hidden_nodes = 10
    n_bias = n_hidden_nodes + env.num_actions
    n_weights = 2 * n_hidden_nodes + n_hidden_nodes * env.num_actions

    # The dimension of a single particle is the number of biases and weights of the neural network
    n = n_bias + n_weights

    num_particles = 100

    # initialize PSO
    pso = PSO(
        n=n
        , num_particles=num_particles
        , fitness_function=lambda weights_and_biases: env.environment_execution(weights_and_biases, n_hidden_nodes)
        , value_bounds=(0, 2)
        , max_iterations=50
        , w=0.5
        , c1=1.5
        , c2=1.5
        , verbose=True
    )
    pso.optimize()
    env.environment_execution(pso.best_particle_population.value, n_hidden_nodes)

    env.render(show_plot=True)
    print("✅ Complete!")

if __name__ == "__main__":
    main()