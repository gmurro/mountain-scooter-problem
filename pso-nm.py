import numpy as np
from Particle import Particle

class InitialPointShapeException(Exception):
    pass


class PSO_NM:
    def __init__(self, n, fitness_function, value_bounds=(0, 2), max_iterations=100, reflection_parameter=1, expansion_parameter=2, contraction_parameter=0.5, shrinkage_parameter=0.5, w=0.5, c1=1.5, c2=1.5, x_1=None, shift_coefficient=1):
        """
        Initialize the Particle Swarm Optimization Nelder-Mead algorithm.
            :param n: Number of dimensions.
            :param fitness_function: Fitness function.
            :param value_bounds: Tuple of the minimum and maximum values for each dimension.
            :param max_iterations: Maximum number of iterations. Default value is 100.
            :param reflection_parameter: Reflection parameter. Default value is 1.
            :param expansion_parameter: Expansion parameter. Default value is 2.
            :param contraction_parameter: Contraction parameter. Default value is 0.5.
            :param shrinkage_parameter: Shrinkage parameter. Default value is 0.5.
            :param w: Inertia weight. Default value is 0.5.
            :param c1: Cognitive parameter. Default value is 1.5.
            :param c2: Social parameter. Default value is 1.5.
            :param x_1: Used as the first point for the simplex generation. Defaults to None, which becomes a random point.
            :param shift_coefficient: Shift coefficient for the simplex initialization. Default value is 1.
        """
        self.n = n
        self.fitness_function = fitness_function
        self.value_bounds = value_bounds
        self.max_iterations = max_iterations
        self.reflection_parameter = reflection_parameter
        self.expansion_parameter = expansion_parameter
        self.contraction_parameter = contraction_parameter
        self.shrinkage_parameter = shrinkage_parameter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.x_1 = x_1
        self.shift_coefficient = shift_coefficient
        self.population = self.initialize_population()

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
        elif self.x_1 is None:  # If the user didn't provide a point
            # Initialize the first point of the simplex randomly
            random_value = np.random.randint(low=self.value_bounds[0], high=self.value_bounds[1]+1, size=self.n)
            first_particle = Particle(0, random_value, self.value_bounds)
        else:  # If the user provided a point, and it is in the right shape
            first_particle = Particle(0, self.x_1, self.value_bounds)

        simplex_particles = [first_particle]

        # Then, we will generate all the other particles
        identity = np.eye(self.n, dtype=int)
        for i in range(self.n):
            # step is positive or negative, to avoid infeasible points
            step = self.shift_coefficient if first_particle.value[i] != self.value_bounds[1] else -self.shift_coefficient
            simplex_particles.append(Particle(i+1, first_particle.value + step * identity[i, :], self.value_bounds))
        return simplex_particles

    def initialize_population(self):
        """
        Initialize the population of particles
            :return: A list of particles representing the population
        """
        # Num of particles in the swarm as defined in the paper
        # by An Liu at al. (A New Hybrid Nelder-Mead Particle Swarm Optimization, 2012)
        num_particles = 2 * self.n + 1

        # the first n+1 particles are constructed using the predetermined starting point and a positive step size of 1.0
        population = self.initialize_simplex()

        # The remaining n particles are randomly generated
        for i in range(self.n+1, num_particles):
            random_value = np.random.randint(low=self.value_bounds[0], high=self.value_bounds[1] + 1, size=self.n)
            population.append(Particle(i, random_value, self.value_bounds))
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
            particle.evaluate_fitness(self.fitness_function)

    def optimize(self):
        """
        Search the optimal solution.
            :return: The optimal solution.
        """
        self.simplex_vals = np.array(
            self.fn(self.simplex_points.transpose()))
        std_dev = np.std(self.simplex_vals)
        i = 0
        while std_dev > target_stddev and i < self.max_iterations:
            self.iterate()
            std_dev = np.std(self.simplex_vals)
            print(
                f"🚀 Performing iteration {i}\t🥴 Standard deviation={round(std_dev, 2)}\t🏅 Value={round(self.min, 3)}")
            i += 1
        if self.fix_result:
            self.fix()
        best, _, _ = self.sort()
        return self.simplex_points[best]


def main():
    pso_nm = PSO_NM(n=3, fitness_function=lambda x: x[0]**2 + x[1]**2, value_bounds=(0, 2))
    print(pso_nm.population)

if __name__ == "__main__":
    main()