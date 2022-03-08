import numpy as np

class Particle:
    def __init__(self, id, value, value_bounds):
        """
        Initialize a particle with a given position and a random velocity
            :param id: Integer representing the identifier of the particle
            :param value: Array representing the position of the particle
            :param value_bounds: Tuple of the minimum and maximum values for each dimension of the particle
        """
        self.id = id
        self.value = value
        self.size = value.size
        self.velocity = np.random.uniform(-1, 1, self.size)
        self.best_value = self.value
        self.best_fitness = -np.inf
        self.fitness = None
        self.value_bounds = value_bounds

    def evaluate(self, fitness_function):
        """
        Evaluate the fitness of the particle
            :param fitness_function: Function to evaluate and update the fitness of the particle
        """
        self.fitness = fitness_function(self.value)

        # check to see if the current position is an individual best
        if self.fitness > self.best_fitness:
            self.best_value = self.value
            self.best_fitness = self.fitness

    def update_velocity(self, w, c1, c2, best_population_value):
        """
        Update the velocity of the particle
              :param w: inertia weight
              :param c1: cognitive weight
              :param c2: social weight
              :param best_population_value: position in the population with highest fitness
        """

        r1 = np.random.random(self.size)
        r2 = np.random.random(self.size)

        cognition_term = c1 * r1 * (self.best_value - self.value)
        social_term = c2 * r2 * (best_population_value - self.value)
        self.velocity = w * self.velocity + cognition_term + social_term

    def update_value(self):
        """
        Update the value (position) of the particle
        """
        self.value = self.value + self.velocity
        self.fix()  # ensure that the value is within the bounds

    def fix(self):
        """
        Fix the value of the particle clipping into the bounds if it is not a feasible solution
            :return: Value of the particle fixed
        """
        self.value = np.rint(self.value)
        self.value = np.clip(self.value, self.value_bounds[0], self.value_bounds[1])

    def __str__(self):
        return "Particle {} (fitness={})".format(self.id, self.fitness)

    def __repr__(self):
        return self.__str__()