import numpy as np
from mountain_scooter import MountainScooter
from Particle import Particle

np.random.seed(71)


class PSO:
    """
    Class for the Particle Swarm Optimization algorithm.
    https://en.wikipedia.org/wiki/Particle_swarm_optimization
    """
    def __init__(self, n, num_particles, fitness_function, max_iterations=50, w=0.4, c1=2.0, c2=2.0, verbose=False):
        """
        Initialize the Particle Swarm Optimization algorithm.
            :param n: Number of dimensions. It represent the size of a single particle.
            :param num_particles: Number of particles in the swarm.
            :param fitness_function: Fitness function to evaluate the particles.
            :param max_iterations: Maximum number of iterations. Default: 50.
            :param w: Inertia weight. Default: 0.4.
            :param c1: Cognitive weight. Default: 2.0.
            :param c2: Social weight. Default: 2.0.
            :param verbose: If True, print the progress of the algorithm. Default value is False.
        """
        self.n = n
        self.num_particles = num_particles
        self.fitness_function = fitness_function
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
            random_value = np.random.random(size=self.n)
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
    print(f"🛵 Starting the MOUNTAIN SCOOTER optimization with PSO algorithm...")

    # initialize environment
    env = MountainScooter(mass=0.4, friction=0.3, max_speed=1.8)

    # The biases have to be the same amount of the nodes without considering the first layer
    # The weights are the connections between the nodes of input and hidden layer + hidden and output layer
    layer_nodes = [2, 8, 7, env.num_actions]
    n_bias = np.sum(layer_nodes) - layer_nodes[0]
    n_weights = 0
    for i in range(0, len(layer_nodes) - 1):
        n_weights += layer_nodes[i] * layer_nodes[i + 1]

    # The dimension of a single particle is the number of biases and weights of the neural network
    n = n_bias + n_weights

    num_particles = 200

    # initialize PSO
    pso = PSO(
        n=n
        , num_particles=num_particles
        , fitness_function=lambda weights_and_biases: env.environment_execution(weights_and_biases, layer_nodes)
        , max_iterations=25
        , w=0.4
        , c1=2.0
        , c2=2.0
        , verbose=True
    )
    pso.optimize()
    env.environment_execution(pso.best_particle_population.value, layer_nodes)

    print(f"\n🏆 Optimal particle: {pso.best_particle_population}")
    env.render(show_plot=True)
    print("✅ Complete!")


if __name__ == "__main__":
    main()