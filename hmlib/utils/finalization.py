"""Run every shutdown hook and preserve failures at the pipeline boundary."""

from typing import Callable, Iterable, Optional, Tuple

from hmlib.log import logger


class FinalizationError(RuntimeError):
    """One or more resources failed to finish their output."""

    def __init__(self, failures: Iterable[Tuple[str, Exception]]) -> None:
        self.failures = tuple(failures)
        super().__init__(
            "Finalization failed: " + "; ".join(f"{name}: {error}" for name, error in self.failures)
        )


def finalize_resources(
    actions: Iterable[Tuple[str, Callable[[], None]]],
    *,
    primary_error: Optional[BaseException] = None,
) -> None:
    """Attempt all hooks, raising on failure unless an earlier error is active.

    Callers in a ``finally`` block pass ``sys.exc_info()[1]`` so shutdown cannot
    replace an exception already escaping the pipeline. In that case every
    cleanup failure is logged with its traceback alongside the original error.
    """
    failures = []
    for name, action in actions:
        try:
            action()
        except Exception as error:
            failures.append((name, error))
    if not failures:
        return
    if primary_error is not None:
        for name, error in failures:
            logger.error(
                "%s failed while handling %s: %s",
                name,
                type(primary_error).__name__,
                primary_error,
                exc_info=(type(error), error, error.__traceback__),
            )
        return
    raise FinalizationError(failures) from failures[0][1]
