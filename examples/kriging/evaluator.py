import importlib.util

from openevolve.evaluation_result import EvaluationResult
import concurrent.futures
import traceback


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

def evaluate(program_path) -> EvaluationResult:
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        if not hasattr(program, "run_ok"):
            print(f"Error: program does not have 'run_ok' function")
        score, rmse_list = run_with_timeout(program.run_ok, timeout_seconds=60)
        # cp_score = conformal_score(geo_uncertainty, upper_bound, lower_bound, y_test)
        return {"RMSE_Cu": rmse_list[0], "RMSE_Pb": rmse_list[1], "RMSE_Zn": rmse_list[2], "combined_score": -score}
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        # return {"coverage": 0.0, "avg_interval_length": 0.0, "error": str(e)}
        return {"RMSE_Cu": 0, "RMSE_Pb": 0, "RMSE_Zn": 0, "combined_score": -10000, "error": str(e)}




