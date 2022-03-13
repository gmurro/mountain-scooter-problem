import numpy as np
from Particle import Particle
from mountain_scooter import MountainScooter

np.random.seed(11)


class InitialPointShapeException(Exception):
    pass


class NM_PSO:
    """
    Class that implement the Nelder-Mead Particle Swarm Optimization algorithm.
    It take inspiration from the paper by An Liu et al. (A New Hybrid Nelder-Mead Particle Swarm Optimization for Coordination Optimization of Directional Overcurrent Relays, 2012).
    """
    def __init__(self, n, fitness_function, max_iterations=50, reflection_parameter=1.0, expansion_parameter=2.0, contraction_parameter=0.5, shrinking_parameter=0.5, w=0.4, c1=2.0, c2=2.0, x_1=None, shift_coefficient=1.0, verbose=False):
        """
        Initialize the Nelder-Mead Particle Swarm Optimization algorithm.
            :param n: Number of dimensions. It represent the size of a single particle.
            :param fitness_function: Fitness function to evaluate the particles.
            :param max_iterations: Maximum number of iterations. Default value is 50.
            :param reflection_parameter: Reflection parameter. Default value is 1.0.
            :param expansion_parameter: Expansion parameter. Default value is 2.0.
            :param contraction_parameter: Contraction parameter. Default value is 0.5.
            :param shrinking_parameter: Shrinkage parameter. Default value is 0.5.
            :param w: Inertia weight. Default value is 0.4.
            :param c1: Cognitive parameter. Default value is 2.0.
            :param c2: Social parameter. Default value is 2.0.
            :param x_1: Used as the first point for the simplex generation. Defaults to None, which becomes a random point.
            :param shift_coefficient: Shift coefficient for the simplex initialization. Default value is 2.
            :param verbose: If True, print the progress of the algorithm. Default value is False.
        """
        self.n = n
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations
        self.reflection_parameter = reflection_parameter
        self.expansion_parameter = expansion_parameter
        self.contraction_parameter = contraction_parameter
        self.shrinking_parameter = shrinking_parameter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.x_1 = x_1
        self.shift_coefficient = shift_coefficient
        self.verbose = verbose
        self.population = self.initialize_population()
        self.best_particle_population = None

    def initialize_simplex(self):
        """
        Initializes the first simplex to begin iterations
            :return: The first simplex points as list of Particle objects.
        """

        # If the user provided a point, and it is not in the right shape
        if isinstance(self.x_1, np.ndarray):
            if len(self.x_1) != self.n:
                raise InitialPointShapeException(
                    f"Please enter an initial point having {self.n} dimensions.")
            else:  # If the user provided a point, and it is in the right shape
                first_particle = Particle(0, self.x_1)
        else:  # If the user didn't provide a point
            # Initialize the first point of the simplex randomly
            random_value = np.random.random(size=self.n)
            first_particle = Particle(0, random_value)

        simplex_particles = [first_particle]

        # Then, we will generate the other particles by shifting the first one
        # in all the position defined by an eye matrix
        identity = np.eye(self.n, dtype=int)
        for i in range(self.n):
            simplex_particles.append(Particle(i+1, first_particle.value + self.shift_coefficient * identity[i, :]))
        return simplex_particles

    def initialize_population(self):
        """
        Initialize the population of particles
            :return: A list of particles representing the population
        """
        # Number of particles in the swarm as defined in the paper
        N = 2 * self.n + 1

        # the first n+1 particles are constructed using the predetermined starting point and a positive step size of 1.0
        population = self.initialize_simplex()

        # The remaining n particles are randomly generated
        for i in range(self.n+1, N):
            random_value = np.random.random(size=self.n)
            population.append(Particle(i, random_value))
        return population

    def sort(self, reverse=True):
        """
        Sort the population by fitness
            :param reverse: If True, sort in descending order. If False, sort in ascending order. Default value is True.
        """
        self.population.sort(key=lambda x: x.fitness, reverse=reverse)

    def evaluate_population(self):
        """
        Evaluate the fitness of each particle in the population
        """
        for particle in self.population:
            particle.evaluate(self.fitness_function)

    def compute_best_particle_population(self):
        """
        Compute the best particle in the population according to the fitness value.
            :return: Particle object representing the best particle in the population.
        """
        return self.population[np.argmax([particle.fitness for particle in self.population])]

    def pso(self):
        """
        Perform one iteration of the PSO algorithm
        """
        for particle in self.population:
            # Update the particle velocity and according to that update the particle value
            particle.update_velocity(self.w, self.c1, self.c2, self.best_particle_population.value)
            particle.update_value()

            # Repair the particle if it is infeasible and evaluate its fitness
            particle.evaluate(self.fitness_function)

    def nelder_mead(self):
        """
        Performs one iteration of the Nelder-Mead method to the top n+1 particles and update the (n+1)th particle:
            - Computes the centroid
            - Tries reflection, expansion, contraction, shrinking
            - Updates the simplex
        If it continuously tries to shrink the simplex, it re-initializes it with the best point
        (https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
        """
        best_particle = self.population[0]
        second_worst_particle = self.population[self.n-1]
        worst_particle = self.population[self.n]

        # Compute the centroid, excluding the worst point
        centroid_value = np.mean([p.value for p in self.population[:self.n]], axis=0)
        centroid_particle = Particle(-1, centroid_value)
        centroid_particle.evaluate(self.fitness_function)

        # REFLECTION
        reflected_value = centroid_value + self.reflection_parameter * (centroid_value - worst_particle.value)
        reflected_particle = Particle(worst_particle.id, reflected_value)
        reflected_particle.evaluate(self.fitness_function)

        # If the new reflected particle is better than the second worst, but worse than the best, we can break to the next iteration
        if best_particle.fitness >= reflected_particle.fitness > second_worst_particle.fitness:
            # then obtain a new simplex by replacing the worst point with the reflected point
            self.population[self.n] = reflected_particle
            if self.verbose:
                print("\t✨ Simplex reflection applied ✨")
            return

        # EXPANSION
        # If the point we've found is better than the best, we try to expand it
        elif reflected_particle.fitness > best_particle.fitness:
            expanded_value = centroid_value + self.expansion_parameter * (reflected_value - centroid_value)
            expanded_particle = Particle(worst_particle.id, expanded_value)
            expanded_particle.evaluate(self.fitness_function)

            # We substitute the worst point with the better of the two
            # If the expanded particle is better than the reflected particle
            if expanded_particle.fitness > reflected_particle.fitness:
                # then obtain a new simplex by replacing the worst point with the expanded point and break to the next iteration
                self.population[self.n] = expanded_particle
                if self.verbose:
                    print("\t✨ Simplex expansion tried, it worked! ✨")
            else:
                # otherwise, we substitute the worst point with the reflected point and break to the next iteration
                self.population[self.n] = reflected_particle
                if self.verbose:
                    print("\t✨ Simplex expansion tried, but reflection was better ✨")
            return

        # CONTRACTION
        # Here it is certain that the reflected point is worse than the second worst
        # If the reflected point we've found was better than the worst, we'll contract
        elif reflected_particle.fitness > worst_particle.fitness:
            contracted_value = centroid_value + self.contraction_parameter * (reflected_value - centroid_value)
            contracted_particle = Particle(worst_particle.id, contracted_value)
            contracted_particle.evaluate(self.fitness_function)

            # If the contracted point is better than the reflected point
            if contracted_particle.fitness > reflected_particle.fitness:
                # then obtain a new simplex by replacing the worst point with the contracted point and break to the next iteration
                self.population[self.n] = contracted_particle

                if self.verbose:
                    print("\t✨ Simplex contraction applied ✨")
                return
        # we will contract too if the reflected point is worse than the worst one
        elif reflected_particle.fitness <= worst_particle.fitness:
            contracted_value = centroid_value + self.contraction_parameter * (worst_particle.value - centroid_value)
            contracted_particle = Particle(worst_particle.id, contracted_value)
            contracted_particle.evaluate(self.fitness_function)

            # If the contracted point is better than the worst point
            if contracted_particle.fitness > worst_particle.fitness:
                # then obtain a new simplex by replacing the worst point with the contracted point and break to the next iteration
                self.population[self.n] = contracted_particle

                if self.verbose:
                    print("\t✨ Simplex contraction applied ✨")
                return

        # SHRINKING
        # If none of the previous methods worked, we'll try our last resort: shrink contraction
        # We'll want to redefine all the simplex points except for the best one.
        for i in range(1, self.n + 1):
            value = best_particle.value + self.shrinking_parameter * (self.population[i].value - best_particle.value)
            self.population[i] = Particle(self.population[i].id, value)
            self.population[i].evaluate(self.fitness_function)

        if self.verbose:
            print("\t✨ Simplex shrinking applied ✨")

    def optimize(self):
        """
        Search the optimal solution.
            :return: The optimal solution.
        """
        # Evaluate the fitness of each particle in the population
        self.evaluate_population()

        for i in range(self.max_iterations):

            # Sort the population by fitness
            self.sort()
            self.best_particle_population = self.population[0]

            if self.verbose:
                print(
                    f"🚀 Performing iteration {i+1}:\n\t📊 "
                    f"Avg={round(np.average([p.fitness for p in self.population]), 2)}\t"
                    f"Best value={self.best_particle_population.fitness}")

            # Apply Nelder-Mead operator to the top n+1 particles and update the (n+1)th particle.
            self.nelder_mead()
            self.best_particle_population = self.compute_best_particle_population()

            # Apply PSO operator for updating the N particles.
            self.pso()
            self.best_particle_population = self.compute_best_particle_population()

        return self.best_particle_population


def main():
    print(f"🛵 Starting the MOUNTAIN SCOOTER optimization with NM-PSO algorithm...")
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

    # initialize NM-PSO
    nm_pso = NM_PSO(n=n
                    , fitness_function=lambda weights_and_biases: env.environment_execution(weights_and_biases, layer_nodes)
                    , max_iterations=20
                    , verbose=True)
    optimal_particle = nm_pso.optimize()
    env.environment_execution(optimal_particle.value, layer_nodes)

    print(f"\n🏆 Optimal particle: {optimal_particle}")
    env.render(show_plot=True)
    print("✅ Complete!")


if __name__ == "__main__":
    main()