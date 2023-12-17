import time
import pandas as pd


class FunctionTimer:
    def __init__(self, funcs, timeout):
        """
        Initialize the FunctionTimer with a list of functions and a timeout.

        Parameters:
        funcs (list): A list of functions to time.
        timeout (float): The timeout for timing the functions.
        """
        self.funcs = funcs
        self.timeout = timeout

    def time_func(self, func):
        """
        Time a function.

        Parameters:
        func (function): The function to time.

        Returns:
        tuple: The number of completions and the mean execution time in milliseconds.
        """
        start = time.time()
        count = 0
        total_time = 0
        while time.time() - start < self.timeout:
            iter_start = time.time()
            func()
            iter_end = time.time()
            total_time += iter_end - iter_start
            count += 1
        mean_time = total_time / count if count != 0 else 0
        return count, mean_time * 1000  # Convert to milliseconds

    def time(self):
        """
        Compare the functions by timing them and printing the results.
        """
        results = []
        for i, func in enumerate(self.funcs, start=1):
            print(f"Timing function {i}...")
            count, mean_time = self.time_func(func)
            results.append(
                {
                    "Function": f"Function {i}",
                    "Completions": count,
                    "Mean Execution Time (ms)": mean_time,
                }
            )

        # Create a DataFrame for the results
        results_df = pd.DataFrame(results)

        print("Timing completed. Here are the results:")
        print(results_df.to_string(index=False))
