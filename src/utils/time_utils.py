import time
from functools import wraps


def format_time(seconds):
    # Handle very small values (less than 1 second)
    if seconds < 1:
        milliseconds = round(seconds * 1000)
        return f"{milliseconds}ms"

    # Handle values less than 1 minute
    if seconds < 60:
        whole_seconds = int(seconds)
        milliseconds = round((seconds - whole_seconds) * 1000)
        return f"{whole_seconds}s {milliseconds}ms" if milliseconds > 0 else f"{whole_seconds}s"

    # Handle values less than 1 hour
    if seconds < 3600:
        minutes, remaining_seconds = divmod(seconds, 60)
        whole_seconds = int(remaining_seconds)
        milliseconds = round((remaining_seconds - whole_seconds) * 1000)
        formatted_time = f"{int(minutes)}m {whole_seconds}s"
        if milliseconds > 0:
            formatted_time += f" {milliseconds}ms"
        return formatted_time

    # Handle values 1 hour or more
    hours, remaining_seconds = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remaining_seconds, 60)
    whole_seconds = int(remaining_seconds)
    milliseconds = round((remaining_seconds - whole_seconds) * 1000)
    formatted_time = f"{int(hours)}h {int(minutes)}m {whole_seconds}s"
    if milliseconds > 0:
        formatted_time += f" {milliseconds}ms"

    return formatted_time


def print_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Function {func.__name__} completed in {format_time(runtime)}")
        return result
    return wrapper
