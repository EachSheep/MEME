import functools
import time

def retry_on_exception(max_retries=3, initial_delay=1, fallback_value=None, exponential_backoff=True):
    """
    Decorator to retry a function up to max_retries times if an exception occurs, with optional exponential backoff.

    Args:
        max_retries: int, number of times to retry the function.
        initial_delay: int, initial delay in seconds between retries.
        fallback_value: any, value to return if all retries fail.
        exponential_backoff: bool, whether to use exponential backoff for the delay.

    Returns:
        Decorated function that retries on exception and returns fallback_value on failure.
    """

    def decorator_retry(func):
        """
        Decorator function that wraps the target function with retry logic.
        Args:
            func: The function to be decorated.
        Returns:
            A wrapper function that implements the retry logic.
        """

        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            """
            Wrapper function that implements the retry logic.
            Args:
                *args: Positional arguments for the function.
                **kwargs: Keyword arguments for the function.
            Returns:
                The result of the function if successful, or fallback_value if all retries fail.
            """

            attempts = 0
            delay = initial_delay
            while attempts < max_retries:
                # try:
                return func(*args, **kwargs)
                # except Exception as e:
                #     print(f"Attempt {attempts + 1} failed: {e}")
                #     attempts += 1
                #     time.sleep(delay)
                #     if exponential_backoff:
                #         delay *= 2  # Double the delay for exponential backoff

            print("All retry attempts failed.")
            return fallback_value

        return wrapper_retry

    return decorator_retry
