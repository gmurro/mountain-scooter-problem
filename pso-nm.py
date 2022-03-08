import numpy as np
from Particle import Particle
from mountain_car import MountainCar

np.random.seed(31)


class InitialPointShapeException(Exception):
    pass


class PSO_NM:
    def __init__(self, n, fitness_function, value_bounds=(0, 2), max_iterations=100, reflection_parameter=1, expansion_parameter=2, contraction_parameter=0.5, shrinking_parameter=0.5, w=0.5, c1=1.5, c2=1.5, x_1=None, shift_coefficient=1, verbose=False):
        """
        Initialize the Particle Swarm Optimization Nelder-Mead algorithm.
            :param n: Number of dimensions.
            :param fitness_function: Fitness function.
            :param value_bounds: Tuple of the minimum and maximum values for each dimension.
            :param max_iterations: Maximum number of iterations. Default value is 100.
            :param reflection_parameter: Reflection parameter. Default value is 1.
            :param expansion_parameter: Expansion parameter. Default value is 2.
            :param contraction_parameter: Contraction parameter. Default value is 0.5.
            :param shrinking_parameter: Shrinkage parameter. Default value is 0.5.
            :param w: Inertia weight. Default value is 0.5.
            :param c1: Cognitive parameter. Default value is 1.5.
            :param c2: Social parameter. Default value is 1.5.
            :param x_1: Used as the first point for the simplex generation. Defaults to None, which becomes a random point.
            :param shift_coefficient: Shift coefficient for the simplex initialization. Default value is 1.
            :param verbose: If True, print the progress of the algorithm. Default value is False.
        """
        self.n = n
        self.fitness_function = fitness_function
        self.value_bounds = value_bounds
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
        elif self.x_1 is None:  # If the user didn't provide a point
            # Initialize the first point of the simplex randomly
            random_value = np.random.randint(low=self.value_bounds[0], high=self.value_bounds[1]+1, size=self.n)
            first_particle = Particle(0, random_value)
        else:  # If the user provided a point, and it is in the right shape
            first_particle = Particle(0, self.x_1)

        simplex_particles = [first_particle]

        # Then, we will generate all the other particles
        identity = np.eye(self.n, dtype=int)
        for i in range(self.n):
            # step is positive or negative, to avoid infeasible points
            step = self.shift_coefficient if first_particle.value[i] != self.value_bounds[1] else -self.shift_coefficient
            simplex_particles.append(Particle(i+1, first_particle.value + step * identity[i, :]))
        return simplex_particles

    def initialize_population(self):
        """
        Initialize the population of particles
            :return: A list of particles representing the population
        """
        # Num of particles in the swarm as defined in the paper
        # by An Liu at al. (A New Hybrid Nelder-Mead Particle Swarm Optimization, 2012)
        N = 2 * self.n + 1

        # the first n+1 particles are constructed using the predetermined starting point and a positive step size of 1.0
        population = self.initialize_simplex()

        # The remaining n particles are randomly generated
        for i in range(self.n+1, N):
            random_value = np.random.randint(low=self.value_bounds[0], high=self.value_bounds[1] + 1, size=self.n)
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

    def repair_population(self):
        """
        Repair the population if any particle is infeasible
        """
        for particle in self.population:
            particle.repair(self.value_bounds)

    def compute_population_best(self):
        return self.population[np.argmax([particle.fitness for particle in self.population])]

    def pso(self):
        """
        Perform one iteration of the PSO algorithm
        """
        for particle in self.population:
            particle.update_velocity(self.w, self.c1, self.c2, self.best_particle_population.value)
            particle.update_value()
            particle.repair(self.value_bounds)
            particle.evaluate(self.fitness_function)

    def nelder_mead(self):
        """
        Performs one iteration of the Nelder-Mead method to the top n+1 particles and update the (n+1)th particle:
            - Computes the centroid
            - Tries reflection, expansion, contraction, shrinking
            - Updates the simplex
        If it continuously tries to shrink the simplex, it re-initializes it with the best point
        """
        best_particle = self.population[0]
        second_worst_particle = self.population[self.n-1]
        worst_particle = self.population[self.n]

        # Compute the centroid, excluding the worst point
        centroid_value = np.mean([p.value for p in self.population[:self.n]], axis=0)
        centroid_particle = Particle(max([p.id for p in self.population])+ 1, centroid_value)
        centroid_particle.repair(self.value_bounds)
        centroid_particle.evaluate(self.fitness_function)

        # REFLECTION
        reflected_value = centroid_value + self.reflection_parameter * (centroid_value - worst_particle.value)
        reflected_particle = Particle(max([p.id for p in self.population])+ 1, reflected_value)
        reflected_particle.repair(self.value_bounds)
        reflected_particle.evaluate(self.fitness_function)

        # If the new reflected particle is better than the second worst, but worse than the best, we can break to the next iteration
        if best_particle.fitness >= reflected_particle.fitness > second_worst_particle.fitness:
            # then obtain a new simplex by replacing the worst point with the reflected point
            self.population[self.n] = reflected_particle
            if self.verbose:
                print("✨ Reflected ✨")
            return

        # EXPANSION
        # If the point we've found is better than the best, we try to expand it
        elif reflected_particle.fitness > best_particle.fitness:
            expanded_value = centroid_value + self.expansion_parameter * (reflected_value - centroid_value)
            expanded_particle = Particle(max([p.id for p in self.population])+ 1, expanded_value)
            expanded_particle.repair(self.value_bounds)
            expanded_particle.evaluate(self.fitness_function)

            # We substitute the worst point with the better of the two
            # If the expanded particle is better than the reflected particle
            if expanded_particle.fitness > reflected_particle.fitness:
                # then obtain a new simplex by replacing the worst point with the expanded point and break to the next iteration
                self.population[self.n] = expanded_particle
                if self.verbose:
                    print("✨ Tried expansion and it worked! ✨")
            else:
                # otherwise, we substitute the worst point with the reflected point and break to the next iteration
                self.population[self.n] = reflected_particle
                if self.verbose:
                    print("✨ Tried expansion but reflection was better ✨")
            return

        # CONTRACTION
        # Here it is certain that the reflected point is worse than the second worst
        # If the reflected point we've found was better than the worst, we'll contract
        elif reflected_particle.fitness > worst_particle.fitness:
            contracted_value = centroid_value + self.contraction_parameter * (reflected_value - centroid_value)
            contracted_particle = Particle(max([p.id for p in self.population])+1, contracted_value)
            contracted_particle.repair(self.value_bounds)
            contracted_particle.evaluate(self.fitness_function)

            # If the contracted point is better than the reflected point
            if contracted_particle.fitness > reflected_particle.fitness:
                # then obtain a new simplex by replacing the worst point with the contracted point and break to the next iteration
                self.population[self.n] = contracted_particle

                if self.verbose:
                    print("✨ Contracted ✨")
                return
        # we will contract too if the reflected point is worse than the worst one
        elif reflected_particle.fitness <= worst_particle.fitness:
            contracted_value = centroid_value + self.contraction_parameter * (worst_particle.value - centroid_value)
            contracted_particle = Particle(max([p.id for p in self.population])+ 1, contracted_value)
            contracted_particle.repair(self.value_bounds)
            contracted_particle.evaluate(self.fitness_function)

            # If the contracted point is better than the worst point
            if contracted_particle.fitness > worst_particle.fitness:
                # then obtain a new simplex by replacing the worst point with the contracted point and break to the next iteration
                self.population[self.n] = contracted_particle

                if self.verbose:
                    print("✨ Contracted ✨")
                return

        # SHRINKING
        # If none of the previous methods worked, we'll try our last resort: shrink contraction
        # We'll want to redefine all the simplex points except for the best one.
        for i in range(1, self.n + 1):
            value = best_particle.value + self.shrinking_parameter * (self.population[i].value - best_particle.value)
            self.population[i] = Particle(self.population[i].id, value)
            self.population[i].repair(self.value_bounds)
            self.population[i].evaluate(self.fitness_function)

        if self.verbose:
            print("✨ Shrinked ✨")

    def optimize(self):
        """
        Search the optimal solution.
            :return: The optimal solution.
        """
        for i in range(self.max_iterations):

            # Evaluate the fitness of each particle in the population
            self.repair_population()
            self.evaluate_population()

            # Sort the population by fitness
            self.sort()
            self.best_particle_population = self.population[0]

            print(
                f"🚀 Performing iteration {i}\t🥴 "
                f"Avg={round(np.average([p.fitness for p in self.population]), 2)}\t"
                f"Best value={self.best_particle_population.fitness}")

            # Apply Nelder-Mead operator to the top n+1 particles and update the (n+1)th particle.
            self.nelder_mead()
            self.best_particle_population = self.compute_population_best()

            # Apply PSO operator for updating the N particles.
            self.pso()
            self.best_particle_population = self.compute_population_best()

        return self.best_particle_population


def evaluate_policy(policy, env, num_bins, exploring_starts=True):
    policy = policy.reshape(num_bins, num_bins)

    # list of thresholds according to which packing in bins the velocity and the position
    velocity_state_array = np.linspace(env.max_speed, env.max_speed, num=num_bins - 1, endpoint=False)
    position_state_array = np.linspace(env.min_position, env.max_position, num=num_bins - 1, endpoint=False)

    # Reset and return the first observation
    velocity, position = env.reset(exploring_starts=exploring_starts)

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


def main():
    env = MountainCar(mass=0.70, friction=0.35, max_speed=2.8)

    num_bins = 20
    pso_nm = PSO_NM(n=num_bins*num_bins
                    , fitness_function=lambda policy: evaluate_policy(policy, env, num_bins, False)
                    , value_bounds=(0, 2)
                    , max_iterations=10
                    , verbose=True)
    optimal_particle = pso_nm.optimize()
    evaluate_policy(optimal_particle.value, env, num_bins, False)
    print("Saving the gif in: ./mountain_car_pso.gif")
    env.render(file_path='./mountain_car_pso.gif', mode='gif')
    print("Complete!")

if __name__ == "__main__":
    main()