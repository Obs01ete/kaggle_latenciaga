import os


def worker_init_fn(worker_id: int) -> None:
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        os.sched_setaffinity(0, range(cpu_count))


def process_init_fn() -> None:
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        os.sched_setaffinity(0, range(cpu_count))
