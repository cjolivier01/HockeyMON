from __future__ import annotations

import errno
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hmlib.utils import output_publication as publication


def _sources(directory):
    directory.mkdir()
    files = {
        "tracking.csv": b"1,7,10,20,30,40\n",
        "camera.csv": b"1,10,20,30,40\n",
        "camera_fast.csv": b"1,11,21,30,40\n",
        "detections.csv": b"",
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)
    return {name: directory / name for name in files}


def _visible(directory):
    return sorted(path.name for path in directory.iterdir() if not path.name.startswith("."))


def _assert_no_staging(directory):
    assert not list(directory.glob(".hm-publish-*"))


def should_publish_complete_independent_copies_with_tracking_last(tmp_path, monkeypatch):
    sources = _sources(tmp_path / "work")
    destination = tmp_path / "game"
    links = []
    original_link = os.link

    def link(source, target, **kwargs):
        assert Path(source).read_bytes() == sources[Path(target).name].read_bytes()
        if Path(target).name == "tracking.csv":
            assert all((destination / name).exists() for name in sources if name != "tracking.csv")
        links.append(Path(target).name)
        original_link(source, target, **kwargs)

    monkeypatch.setattr(publication.os, "link", link)
    result = publication.publish_artifacts(sources, destination, exact=True)
    assert result.suffix == 0
    assert links[-1] == "tracking.csv"
    for name, source in sources.items():
        assert result.files[name].read_bytes() == source.read_bytes()
        assert result.files[name].stat().st_ino != source.stat().st_ino
        assert result.files[name].stat().st_mode == source.stat().st_mode
    sources["tracking.csv"].write_bytes(b"changed")
    assert result.files["tracking.csv"].read_bytes() != b"changed"
    _assert_no_staging(destination)


def should_choose_generation_above_existing_companion_suffix(tmp_path):
    sources = _sources(tmp_path / "work")
    destination = tmp_path / "game"
    destination.mkdir()
    (destination / "camera-12.csv").write_bytes(b"old")
    result = publication.publish_artifacts(sources, destination)
    assert result.suffix == 13
    assert result.files["tracking.csv"].name == "tracking-13.csv"
    assert (destination / "camera-12.csv").read_bytes() == b"old"


def should_treat_dangling_symlink_as_occupied_generation(tmp_path):
    sources = _sources(tmp_path / "work")
    destination = tmp_path / "game"
    destination.mkdir()
    (destination / "camera.csv").symlink_to("missing")
    result = publication.publish_artifacts(sources, destination)
    assert result.suffix == 1
    assert (destination / "camera.csv").is_symlink()


def should_serialize_concurrent_generation_reservations(tmp_path):
    sources = _sources(tmp_path / "work")
    destination = tmp_path / "game"
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(lambda _: publication.publish_artifacts(sources, destination), range(3))
        )
    assert sorted(result.suffix for result in results) == [1, 2, 3]
    for result in results:
        assert len(result.files) == 4
        assert all(path.exists() for path in result.files.values())
    _assert_no_staging(destination)


@pytest.mark.parametrize("failure_stage", ["copy", "fsync", "fsync_after_marker", "link"])
def should_rollback_partial_publication_and_retain_working_files(
    tmp_path, monkeypatch, failure_stage
):
    sources = _sources(tmp_path / "work")
    destination = tmp_path / "game"
    if failure_stage == "copy":

        def fail_copy(source, target, name):
            target.write(b"incomplete")
            raise OSError(errno.ENOSPC, "copy full")

        monkeypatch.setattr(publication, "_copy_complete", fail_copy)
    elif failure_stage.startswith("fsync"):
        original_sync = os.fsync
        calls = []

        def fail_sync(descriptor):
            calls.append(descriptor)
            if len(calls) == (6 if failure_stage == "fsync_after_marker" else 5):
                raise OSError(errno.EIO, "directory fsync failed")
            original_sync(descriptor)

        monkeypatch.setattr(publication.os, "fsync", fail_sync)
    else:
        original_link = os.link

        def fail_link(source, target, **kwargs):
            if Path(target).name == "tracking.csv":
                raise OSError(errno.ENOSPC, "link full")
            original_link(source, target, **kwargs)

        monkeypatch.setattr(publication.os, "link", fail_link)
    with pytest.raises(OSError):
        publication.publish_artifacts(sources, destination, exact=True)
    assert _visible(destination) == []
    assert all(path.exists() for path in sources.values())
    _assert_no_staging(destination)


def should_handle_nfs_link_error_after_successful_link(tmp_path, monkeypatch):
    sources = _sources(tmp_path / "work")
    original_link = os.link

    def ambiguous_link(source, target, **kwargs):
        original_link(source, target, **kwargs)
        raise OSError(errno.EIO, "NFS response lost")

    monkeypatch.setattr(publication.os, "link", ambiguous_link)
    result = publication.publish_artifacts(sources, tmp_path / "game")
    assert all(path.exists() for path in result.files.values())


def should_never_overwrite_an_exact_destination(tmp_path):
    sources = _sources(tmp_path / "work")
    destination = tmp_path / "game"
    destination.mkdir()
    (destination / "camera.csv").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        publication.publish_artifacts(sources, destination, exact=True)
    assert _visible(destination) == ["camera.csv"]
    assert (destination / "camera.csv").read_bytes() == b"existing"


def should_publish_suffixed_tracking_after_companions(tmp_path, monkeypatch):
    sources = _sources(tmp_path / "work")
    sources = {publication.artifact_name(name, 7): path for name, path in sources.items()}
    original_link = os.link
    links = []

    def record_link(source, target, **kwargs):
        links.append(Path(target).name)
        original_link(source, target, **kwargs)

    monkeypatch.setattr(publication.os, "link", record_link)
    publication.publish_artifacts(sources, tmp_path / "game", exact=True)
    assert links[-1] == "tracking-7.csv"


def should_deploy_video_and_csvs_as_one_generation(tmp_path):
    from hmlib.cli.hmtrack import _deploy_output_artifacts

    sources = _sources(tmp_path / "work")
    video = tmp_path / "work" / "tracking_output-with-audio.mp4"
    video.write_bytes(b"video")
    destination = tmp_path / "game"
    destination.mkdir()
    (destination / "camera-2.csv").write_bytes(b"old")
    published_video = _deploy_output_artifacts(
        output_video_path=str(video),
        output_video=None,
        results_folder=str(video.parent),
        target_deploy_dir=str(destination),
        game_id="game",
    )
    assert published_video.name == "game-tracking_output-with-audio-3.mp4"
    assert published_video.read_bytes() == b"video"
    assert (destination / "tracking-3.csv").read_bytes() == sources["tracking.csv"].read_bytes()


def should_preserve_explicit_video_name_and_matching_csv_suffix(tmp_path):
    from hmlib.cli.hmtrack import _deploy_output_artifacts

    _sources(tmp_path / "work")
    video = tmp_path / "work" / "output.mp4"
    video.write_bytes(b"video")
    destination = tmp_path / "game"
    explicit = destination / "my-movie-4.mp4"
    published_video = _deploy_output_artifacts(
        output_video_path=str(video),
        output_video=str(explicit),
        results_folder=str(video.parent),
        target_deploy_dir=str(destination),
        game_id="game",
    )
    assert published_video == explicit
    assert (destination / "tracking-4.csv").exists()
    with pytest.raises(FileExistsError):
        _deploy_output_artifacts(
            output_video_path=str(video),
            output_video=str(explicit),
            results_folder=str(video.parent),
            target_deploy_dir=str(destination),
            game_id="game",
        )


def should_reject_symlinked_source_artifacts(tmp_path):
    source = tmp_path / "link.csv"
    original = tmp_path / "original.csv"
    original.write_bytes(b"private")
    source.symlink_to(original)
    destination = tmp_path / "game"
    with pytest.raises(OSError):
        publication.publish_artifacts({"tracking.csv": source}, destination)
    assert _visible(destination) == []
    _assert_no_staging(destination)


def should_preserve_replaced_files_during_failed_publication(tmp_path, monkeypatch, caplog):
    sources = _sources(tmp_path / "work")
    destination = tmp_path / "game"
    original_link = os.link

    def replace_companion(source, target, **kwargs):
        if Path(target).name == "tracking.csv":
            camera = destination / "camera.csv"
            camera.unlink()
            camera.write_bytes(b"new owner")
            raise OSError("tracking link failed")
        original_link(source, target, **kwargs)

    monkeypatch.setattr(publication.os, "link", replace_companion)
    with pytest.raises(OSError, match="tracking link failed"):
        publication.publish_artifacts(sources, destination, exact=True)
    assert _visible(destination) == ["camera.csv"]
    assert (destination / "camera.csv").read_bytes() == b"new owner"
    assert "changed ownership" in caplog.text
    _assert_no_staging(destination)


@pytest.mark.parametrize(
    "history,expected",
    [
        ([], 1),
        (["game-tracking_output-with-audio.mp4"], 1),
        (
            [
                "game-tracking_output-with-audio-1.mp4",
                "game-tracking_output-with-audio-3.mp4",
                "game-stitched_output-with-audio-3.mp4",
                "game-stitched_output-with-audio-4.mp4",
            ],
            5,
        ),
        (["stitched_output-1001.mkv"], 1002),
        (["rink_mask_0-17.png"], 18),
    ],
)
def should_number_videos_csvs_and_run_mask_above_both_video_histories(tmp_path, history, expected):
    from hmlib.cli.hmtrack import _deploy_output_artifacts

    sources = _sources(tmp_path / "work")
    video = tmp_path / "work" / "tracking_output-with-audio.mp4"
    video.write_bytes(b"video")
    mask = tmp_path / "work" / "rink_mask_0.png"
    mask.write_bytes(b"the mask used for this run")
    destination = tmp_path / "game"
    destination.mkdir()
    for name in history:
        (destination / name).write_bytes(b"preserved")
    published = _deploy_output_artifacts(
        output_video_path=str(video),
        output_video=None,
        results_folder=str(video.parent),
        target_deploy_dir=str(destination),
        game_id="game",
    )
    assert published.name == f"game-tracking_output-with-audio-{expected}.mp4"
    for name, source in sources.items():
        assert (
            destination / publication.artifact_name(name, expected)
        ).read_bytes() == source.read_bytes()
    mask.write_bytes(b"next run mask")
    assert (
        destination / f"rink_mask_0-{expected}.png"
    ).read_bytes() == b"the mask used for this run"
    for name in history:
        assert (destination / name).read_bytes() == b"preserved"


@pytest.mark.parametrize("separate_video_directory", [False, True])
def should_number_unnumbered_explicit_output_without_overwriting_calibration(
    tmp_path, separate_video_directory
):
    from hmlib.cli.hmtrack import _deploy_output_artifacts

    _sources(tmp_path / "work")
    video = tmp_path / "work" / "output.mp4"
    video.write_bytes(b"video")
    (video.parent / "rink_mask_0.png").write_bytes(b"run mask")
    game = tmp_path / "game"
    game.mkdir()
    (game / "rink_mask_0.png").write_bytes(b"current calibration")
    (game / "game-stitched_output-with-audio-4.mp4").write_bytes(b"old video")
    video_dir = tmp_path / "movies" if separate_video_directory else game
    published = _deploy_output_artifacts(
        output_video_path=str(video),
        output_video=str(video_dir / "custom.mp4"),
        results_folder=str(video.parent),
        target_deploy_dir=str(game),
        game_id="game",
    )
    assert published == video_dir / "custom-5.mp4"
    assert published.read_bytes() == b"video"
    assert (game / "tracking-5.csv").exists()
    assert (game / "rink_mask_0-5.png").read_bytes() == b"run mask"
    assert (game / "rink_mask_0.png").read_bytes() == b"current calibration"
