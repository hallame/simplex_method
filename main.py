import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import linprog


class SimplexMethod:
    def __init__(self):
        pass

    def solve(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='simplex'):
        """
        Solves the linear programming problem using the simplex method.

        Parameters:
            c: Coefficients of the objective function.
            A_ub: 2D array of inequality constraint coefficients.
            b_ub: 1D array of inequality constraint upper-bound values.
            A_eq: 2D array of equality constraint coefficients.
            b_eq: 1D array of equality constraint values.
            bounds: List of variable bounds (min, max) for each variable.
            method: Method to use ('simplex').
        Returns:
            A dict containing the solution, status, message, and other information.
        """
        start_time = time.time()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
        end_time = time.time()

        print(f"Computation Time: {end_time - start_time:.6f} seconds")
        return res

    def visualize_2d(self, A_ub, b_ub, c, res):
        """
        Visualizes the feasible region and the objective function for a 2D problem.
        """
        # Plot the constraints
        x = np.linspace(0, 10, 500)
        for i in range(len(A_ub)):
            plt.plot(x, (b_ub[i] - A_ub[i][0] * x) / A_ub[i][1], label=f'Constraint {i + 1}')

        # Plot the objective function
        plt.plot(x, (res.fun - c[0] * x) / c[1], label='Objective Function', color='red', linestyle='--')

        # Highlight the solution
        plt.scatter(res.x[0], res.x[1], color='red', zorder=5, label='Optimal Solution')

        plt.xlim((0, 10))
        plt.ylim((0, 10))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.legend()
        plt.title('2D Simplex Visualization')
        plt.show()


# Example usage and comparison
def compare_with_scipy():
    solver = SimplexMethod()

    # Define problems for testing (both min and max)
    examples = [
        # Example 1: Minimize
        {
            'c': [1, 2],
            'A_ub': [[-1, 1], [1, 1], [1, -2]],
            'b_ub': [1, 3, 1],
            'A_eq': None,
            'b_eq': None,
            'bounds': [(0, None), (0, None)]
        },
        # Example 2: Maximize (reversed sign on objective)
        {
            'c': [-1, -2],
            'A_ub': [[-1, 1], [1, 1], [1, -2]],
            'b_ub': [1, 3, 1],
            'A_eq': None,
            'b_eq': None,
            'bounds': [(0, None), (0, None)]
        },
        # Example 3: Minimize with different constraints
        {
            'c': [3, 2],
            'A_ub': [[2, 1], [1, 1], [-1, 1]],
            'b_ub': [4, 2, 1],
            'A_eq': None,
            'b_eq': None,
            'bounds': [(0, None), (0, None)]
        },
        # Example 4: Maximize with equalities
        {
            'c': [-3, -4],
            'A_ub': [[1, 0], [0, 1]],
            'b_ub': [5, 6],
            'A_eq': [[1, 1]],
            'b_eq': [8],
            'bounds': [(0, None), (0, None)]
        },
        # Example 5: More complex Minimize
        {
            'c': [2, 5],
            'A_ub': [[1, 2], [2, 1]],
            'b_ub': [20, 16],
            'A_eq': None,
            'b_eq': None,
            'bounds': [(0, None), (0, None)]
        }
    ]

    for i, example in enumerate(examples):
        print(f"\nExample {i + 1}:")
        print("Custom Simplex Method:")
        res_custom = solver.solve(**example)
        print(res_custom)

        print("SciPy Simplex Method:")
        start = time.time()
        res_scipy = linprog(**example, method='simplex')
        end = time.time()
        print(f"Computation Time: {end - start:.6f} seconds")
        print(res_scipy)

        # Visualize for 2D problems
        if len(example['c']) == 2:
            solver.visualize_2d(example['A_ub'], example['b_ub'], example['c'], res_custom)

# Run comparisons
compare_with_scipy()

