import time
from typing import Dict, List


class StopWatchNotStarted(Exception):
    """Exception raised when an operation is attempted on a stopwatch that hasn't been started."""

    def __init__(self, message="Stopwatch has not been started yet"):
        self.message = message
        super().__init__(self.message)


class StopWatch:

    def __init__(self) -> None:
        self.total = {}
        self.started_time = {}

    def start(self, name: str) -> None:
        if name in self.started_time:
            return None

        self.started_time[name] = time.perf_counter()

    def stop(self, name: str) -> None:

        stop_time = time.perf_counter()

        if name not in self.started_time:
            raise StopWatchNotStarted()

        time_elapsed = stop_time - self.started_time.pop("name")

        if not name in self.total:
            self.total[name] = 0

        self.total[name] = self.total[name] + time_elapsed

    def reset(self, name: str) -> None:
        self.total.pop(name)

    def get_total(self, name: str) -> float:
        return self.total.get(name)

    def get_all_total(self) -> Dict[str, float]:
        return self.total


class Timer:
    def __init__(self, stopwatch: StopWatch, name: str) -> None:
        self.stopwatch = stopwatch
        self.name = name

    def __enter__(self):
        self.stopwatch.start(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stopwatch.stop(self.name)


def merge_stopwatch_data(stopwatches: List[StopWatch]) -> Dict[str, float]:
    dicts = [s.get_all_total() for s in stopwatches]
    return merge_stopwatch_dict(dicts)


def merge_stopwatch_dict(dicts: List[Dict[str, float]]):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
    return result
