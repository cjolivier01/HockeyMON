"""Serialize OpenMMLab's process-global registry scope during model setup."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

_REGISTRY_LOCK = RLock()


@contextmanager
def model_registry_scope(scope: str) -> Iterator[None]:
    """Keep concurrent factories in their own scope and restore the caller's scope.

    MMEngine's DefaultScope is process-global, including its temporary overrides.
    Hold one reentrant lock across registry construction and checkpoint loading;
    restoring a scope alone still races with another factory's active override.
    Ordinary model execution does not need this guard unless it builds a lazy
    transform pipeline, as MMDetection's segmentation inference helper does.
    """
    from mmengine.registry import DefaultScope

    with _REGISTRY_LOCK, DefaultScope.overwrite_default_scope(scope):
        yield
