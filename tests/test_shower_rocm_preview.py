from __future__ import annotations

from unittest import mock

import hmlib.ui.shower as shower_module


def should_report_native_cuda_preview_unavailable_on_rocm(monkeypatch) -> None:
    monkeypatch.setattr(shower_module, "show_cuda_tensor", object())
    monkeypatch.setattr(shower_module, "_is_rocm_runtime", lambda: True)

    assert shower_module._native_cuda_preview_available() is False


def should_report_native_cuda_preview_available_without_rocm(monkeypatch) -> None:
    monkeypatch.setattr(shower_module, "show_cuda_tensor", object())
    monkeypatch.setattr(shower_module, "_is_rocm_runtime", lambda: False)

    assert shower_module._native_cuda_preview_available() is True


def should_report_opencv_cuda_preview_available_without_hockeymon_extension(
    monkeypatch,
) -> None:
    monkeypatch.setattr(shower_module, "_is_rocm_runtime", lambda: False)

    assert shower_module._opencv_cuda_preview_available() is True


def should_warn_once_when_local_preview_runs_on_rocm(monkeypatch) -> None:
    monkeypatch.setattr(shower_module, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(shower_module, "_native_cuda_preview_available", lambda: False)
    monkeypatch.setattr(shower_module, "has_local_display", lambda: True)
    monkeypatch.setattr(shower_module, "create_queue", lambda mp=False: mock.Mock())

    fake_thread = mock.Mock()
    fake_thread.start = mock.Mock()
    monkeypatch.setattr(shower_module.threading, "Thread", lambda target: fake_thread)

    logger = mock.Mock()
    shower = shower_module.Shower(label="preview", logger=logger)

    logger.warning.assert_called_once()
    shower._thread = None
