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
    def __init__(self, n, num_particles, fitness_function, value_bounds=(0, 2),  max_iterations=50, w=0.5, c1=1.5, c2=1.5, verbose=False):
        """
        Initialize the Particle Swarm Optimization algorithm.
            :param n: Number of dimensions. It represent the size of a single particle.
            :param num_particles: Number of particles in the swarm.
            :param fitness_function: Fitness function to evaluate the particles.
            :param value_bounds: Tuple with the minimum and maximum values for each dimension. Default value is (0, 2).
            :param max_iterations: Maximum number of iterations. Default: 50.
            :param w: Inertia weight. Default: 0.5.
            :param c1: Cognitive weight. Default: 1.5.
            :param c2: Social weight. Default: 1.5.
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
                    f"🚀 Performing iteration {i}:\n\t📊 "
                    f"Avg={round(np.average([p.fitness for p in self.population]), 2)}\t"
                    f"Best value={self.best_particle_population.fitness}")

            for particle in self.population:
                # Update the particle velocity and according to that update the particle value
                particle.update_velocity(self.w, self.c1, self.c2, self.best_particle_population.value)
                particle.update_value()

                # Repair the particle if it is infeasible and evaluate its fitness
                particle.repair(self.value_bounds)
                particle.evaluate(self.fitness_function)
            self.best_particle_population = self.compute_best_particle_population()



def main():
    # initialize environment
    env = MountainScooter(mass=0.70, friction=0.35, max_speed=2.8)

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