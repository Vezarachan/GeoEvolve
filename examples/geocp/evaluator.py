import numpy as np
import concurrent.futures
import traceback
import signal
import importlib.util
from numpy.typing import NDArray


def interval_score(y_true, lower_bound, upper_bound, alpha=0.1, epsilon=1e-6, normalize=True, lambda_penalty=1.0):
    width = np.maximum(upper_bound - lower_bound, epsilon)
    below = (lower_bound - y_true) * (y_true < lower_bound)
    above = (y_true - upper_bound) * (y_true > upper_bound)
    if normalize:
        scale = np.maximum(np.abs(y_true), 1.0)
        width = width / scale
        below = below / scale
        above = above / scale
    score = width + lambda_penalty * (2 / alpha) * (below + above)
    score = np.where(np.isnan(score), 0.0, score)
    return np.mean(score)

def conformal_score(geo_uncertainty, upper_bound, lower_bound, y_test, epsilon=0.1):
    coverage = compute_coverage(y_test, upper_bound, lower_bound)
    avg_interval_length = average_interval_length(geo_uncertainty)
    tau = 1 - epsilon
    score = avg_interval_length / tau * (1 + np.square(np.maximum(0, tau - coverage) / tau))
    return score

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

def compute_coverage(y_test: NDArray, upper_bound: NDArray, lower_bound: NDArray):
    return np.mean((y_test >= lower_bound) & (y_test <= upper_bound))

def average_interval_length(geo_uncertainty: NDArray):
    return np.mean(geo_uncertainty)

def run_validation(geo_uncertainty, upper_bound, lower_bound, y_test):
    coverage = compute_coverage(y_test, upper_bound, lower_bound)
    avg_interval_length = average_interval_length(geo_uncertainty)

    return coverage, avg_interval_length

def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """

    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        if not hasattr(program, "run_geocp"):
            print(f"Error: program does not have 'run_geocp' function")
        geo_uncertainty, upper_bound, lower_bound, y_test = run_with_timeout(program.run_geocp, timeout_seconds=5)
        coverage, avg_interval_length = run_validation(geo_uncertainty, upper_bound, lower_bound, y_test)
        # cp_score = conformal_score(geo_uncertainty, upper_bound, lower_bound, y_test)
        score = interval_score(y_test, lower_bound, upper_bound)
        return {"interval_score": score, "coverage": coverage, "avg_interval_length": avg_interval_length, "combined_score": -score}
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        # return {"coverage": 0.0, "avg_interval_length": 0.0, "error": str(e)}
        return {"interval_score": 0.0, "coverage": 0.0, "avg_interval_length": 0.0, "combined_score": -1000, "error": str(e)}

