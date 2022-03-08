import numpy as np
from enum import Enum

class InitialPointShapeException(Exception):
    pass


class NoSimplexDefinedException(Exception):
    pass


class Operations(Enum):
    REFLECTION = 0
    EXPANSION = 1
    CONTRACTION = 2
    SHRINK = 3


class NelderMead:
    def __init__(self, n, fn, reflection_parameter=1, expansion_parameter=2, contraction_parameter=0.5, shrinkage_parameter=0.5, max_iterations=50, shift_coefficient=0.05, verbose=False):
        """
        Initializes the optimizer
            :param n: Number of variables
            :param fn: Objective function to minimize
            :param reflection_parameter: Reflection coefficient. Defaults to 1.
            :param expansion_parameter: Expansion coefficient. Defaults to 2.
            :param contraction_parameter: Contraction coefficient. Defaults to 0.5.
            :param shrinkage_parameter: Shrinkage coefficient. Defaults to 0.5.
            :param max_iterations: Limit of iterations in optimization. Defaults to 50.
            :param shift_coefficient: Coefficient of shift in the initial points. Defaults to 0.05.
            :param verbose: If True, the algorithm outputs the steps while they are made. Defaults to False.
            :param fix_result: Fixes the result to the sum constraint. Defaults to True.
        """
        self.reflection_parameter = reflection_parameter
        self.expansion_parameter = expansion_parameter
        self.contraction_parameter = contraction_parameter
        self.shrinkage_parameter = shrinkage_parameter
        self.n = n
        self.fn = fn
        self.max_iterations = max_iterations
        self.last_performed_operation = None
        self.shift_coefficient = shift_coefficient
        self.verbose = verbose
        self.fix_result = fix_result

    def initialize_simplex(self, x_1=None):
        """
        Initializes the first simplex to begin iterations
            :param x_1: used as the first point for the simplex generation. Defaults to None, which becomes a random point.
            :raise InitialPointShapeException: Raised when the provided first point has the wrong number of dimensions.
        """
        self.simplex_points = np.empty((self.n+1, self.n))
        # If the user provided a point, and it is not in the right shape
        if isinstance(x_1, np.ndarray):
            if len(x_1) != self.n:
                raise InitialPointShapeException(
                    f"Please enter an initial point having {self.n} dimensions.")
        elif x_1 == None:  # If the user didn't provide a point
            self.simplex_points[0] = np.random.rand(self.n)
        else:  # If the user provided a point, and it is in the right shape
            self.simplex_points[0] = x_1
        # Then, we will generate all the other points
        for i in range(1, self.n+1):  # The simplex has n+1 points
            shift_coefficient_i = self.shift_coefficient if self.simplex_points[0][i -
                                                                                   1] != 0 else self.shift_coefficient/2
            unit_vector_i = np.zeros(self.n)
            unit_vector_i[i-1] = 1
            self.simplex_points[i] = self.simplex_points[0] + \
                shift_coefficient_i * unit_vector_i
        print(f"Succesfully initialized simplex: {self.simplex_points}")

    def sort(self):
        """
        Fills self.simplex_points with the function values, then
        returns the best, second worst and worst points.
        Returns:
            - tuple: Best, second worst and worst indices of the simplex points' values
        """
        # Calculate values of the function in all points of the simplex
        self.simplex_vals = np.array(
            self.fn(self.simplex_points.transpose()))
        sorted_indices = np.argsort(self.simplex_vals)
        self.min = self.simplex_vals[sorted_indices[0]]
        return sorted_indices[0], sorted_indices[-2], sorted_indices[-1]

    def iterate(self):
        """Performs one iteration of the Nelder-Mead method:
            - Sorts the simplex points
            - Computes the centroid
            - Tries reflection, expansion, contraction, shrinking
            - Updates the simplex
            If it continuously tries to shrink the simplex, it re-initializes it with the best point
        """
        best, sec_worst, worst = self.sort()
        # Compute the centroid, excluding the worst point
        centroid = np.mean(np.delete(self.simplex_points, worst, 0), axis=0)
        # Transformation: reflection
        x_reflected = centroid + \
            (self.reflection_parameter * (centroid-self.simplex_points[worst]))
        y_reflected = self.fn(x_reflected)
        # If the new point is better than the second worst, but worse than the best, we can break to the next iteration
        if self.simplex_vals[best] < y_reflected <= self.simplex_vals[sec_worst]:
            # We don't want negative points
            self.simplex_points[worst] = x_reflected
            # Substitute negative values with 0
            self.simplex_points[worst][self.simplex_points[worst] < 0] = 0
            self.last_performed_operation = Operations.REFLECTION
            if self.verbose:
                print("✨ Reflected ✨")
            return
        # If the point we've found is better than the best, we try to expand it
        elif y_reflected < self.simplex_vals[best]:
            x_expanded = centroid + self.expansion_parameter * \
                (x_reflected-centroid)
            y_expanded = self.fn(x_expanded)
            # We substitute the worst point with the better of the two
            if y_expanded < y_reflected:
                self.simplex_points[worst] = x_expanded
                # Substitute negative values with 0
                self.simplex_points[worst][self.simplex_points[worst] < 0] = 0
                if self.verbose:
                    print("✨ Tried expansion and it worked! ✨")
                self.last_performed_operation = Operations.EXPANSION
            else:
                self.simplex_points[worst] = x_reflected
                # Substitute negative values with 0
                self.simplex_points[worst][self.simplex_points[worst] < 0] = 0
                if self.verbose:
                    print("✨ Tried expansion but reflection was better ✨")
                self.last_performed_operation = Operations.REFLECTION
            return
        # If the point we've found was worse than the second worst, we'll contract
        elif y_reflected > self.simplex_vals[sec_worst]:
            x_contracted = centroid + self.contraction_parameter * \
                (self.simplex_points[worst] - centroid)
            y_contracted = self.fn(x_contracted)
            if y_contracted < self.simplex_vals[worst]:
                self.simplex_points[worst] = x_contracted
                # Substitute negative values with 0
                self.simplex_points[worst][self.simplex_points[worst] < 0] = 0
                self.last_performed_operation = Operations.CONTRACTION
                if self.verbose:
                    print("✨ Contracted ✨")
                return
        # If none of the previous methods worked, we'll try our last resort: shrink contraction
        # We'll want to redefine all the simplex points except for the best one.
        for i in range(self.n+1):
            if (i != best):  # We won't change the best one
                self.simplex_points[i] = self.simplex_points[best] + self.shrinkage_parameter * (
                    self.simplex_points[i] - self.simplex_points[best])
                # Substitute negative values with 0
                #self.simplex_points[i][self.simplex_points[i] < 0] = 0
        self.last_performed_operation = Operations.SHRINK
        if self.verbose:
            print("✨ Shrinked ✨")

    def fix(self):
        """Reduces the simplex points' size to satisfy the constraint
        """
        self.simplex_points = (
            self.simplex_points / np.sum(self.simplex_points, axis=1, keepdims=1)) * self.sum_constraint

    def fit(self, target_stddev):
        """Computes until the STD deviation of the function values in the simplex reaches a given value
        Args:
            target_stddev (float, optional): Target standard deviation
        Returns:
            tuple: point of maximum X and its value
        """
        # Check if simplex points has been defined, i.e. initialize_simplex has been called
        if type(self.simplex_points) is not np.ndarray:
            raise NoSimplexDefinedException
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


if __name__ == '__main__':
    def fn(x): return ((x[0]+2*x[1]-7)**2 + (2*x[0]+x[1]-5)**2)
    nm = NelderMead(2, fn, fix_result=False)
    nm.initialize_simplex()
    print(nm.fit(0.00001))
    text = "as seen on WolframAlpha."
    link = "https://www.wolframalpha.com/input/?i=minimize+%28x_0+plus+2*x_1-7%29**2+plus++++%282*x_0+plus+x_1-5%29**2"
    print(
        f"The result should be (1,3), \u001b]8;;{link}\u001b\\{text}\u001b]8;;\u001b\\")