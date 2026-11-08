"""Tiny helpers shared by all modules."""
import timeit
import tracemalloc


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show(label: str, value) -> None:
    print(f"  {label:<34} {value}")


def bench(fn, repeat: int = 3) -> float:
    """Best-of-`repeat` seconds per call (auto-scaled number of loops)."""
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeat, number=number)) / number


def peak_mem(fn) -> int:
    """Peak bytes allocated while running fn (NumPy reports to tracemalloc)."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak
