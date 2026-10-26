"""Bounded selected JPEG snapshots for the private web Program supervisor.

The frame owner holds its route lock across publication. Readers verify the
manifest and pinned JPEG bytes before decoding; retired snapshots are retried.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from hmlib.ui.hstream_web import PinnedFile, canonical_bytes, worker_binding

PROTOCOL = "hstream-web-preview"
MANIFEST = "web-preview.json"
OWNER = "web-preview-owner.json"
MAX_MANIFEST_BYTES = 16 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 4 * 1024 * 1024
MAX_SOURCE_SIDE = 131072
CHANNELS = {
    "Final": "program",
    "Stitched": "stitched",
    "Camera 1": "source0",
    "Camera 2": "source1",
}


def _write_ephemeral(path: Path, document: dict) -> None:
    data = canonical_bytes(document)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Selected preview metadata exceeds its byte limit")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".web-preview-", delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _positive(value, maximum: int) -> bool:
    return type(value) is int and 0 < value <= maximum


def read_owner(session: Path, *, expected: dict) -> dict:
    """Applied route ownership; the outer supervisor separately proves liveness."""
    pinned = PinnedFile(session, OWNER)
    try:
        document = json.loads(pinned.read(MAX_MANIFEST_BYTES))
    finally:
        pinned.close()
    if not isinstance(document, dict) or set(document) != {
        "protocol",
        "version",
        "run_id",
        "request_digest",
        "session_id",
        "channel",
        "generation",
        "status",
    }:
        raise ValueError("Invalid selected preview owner fields")
    if (
        document["protocol"] != "hstream-web-preview-owner"
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        raise ValueError("Unsupported selected preview owner")
    if document["status"] not in {"selected", "retired"} or document["channel"] not in {
        *CHANNELS.values(),
        "none",
    }:
        raise ValueError("Invalid selected preview ownership")
    if not _positive(document["generation"], 2**53 - 1):
        raise ValueError("Invalid selected preview owner generation")
    import uuid

    if (
        not isinstance(document["run_id"], str)
        or str(uuid.UUID(document["run_id"])) != document["run_id"]
    ):
        raise ValueError("Invalid selected preview owner run")
    for key, length in (("request_digest", 64), ("session_id", 32)):
        if not isinstance(document[key], str) or not re.fullmatch(
            f"[0-9a-f]{{{length}}}", document[key]
        ):
            raise ValueError("Invalid selected preview owner binding")
    if any(
        type(document.get(key)) is not type(value) or document.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("Selected preview owner belongs to another route or run")
    return document


def retire_owner(session: Path, *, expected: dict) -> None:
    try:
        owner = read_owner(session, expected=expected)
    except FileNotFoundError:
        (session / MANIFEST).unlink(missing_ok=True)
        return
    (session / MANIFEST).unlink(missing_ok=True)
    _write_ephemeral(session / OWNER, {**owner, "status": "retired"})


def validate_manifest(document: dict) -> None:
    if not isinstance(document, dict) or set(document) != {
        "protocol",
        "version",
        "run_id",
        "request_digest",
        "session_id",
        "channel",
        "generation",
        "sequence",
        "frame",
        "source",
        "image",
        "overlays",
    }:
        raise ValueError("Invalid selected preview manifest fields")
    if (
        document["protocol"] != PROTOCOL
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        raise ValueError("Unsupported selected preview manifest")
    import uuid

    if (
        not isinstance(document["run_id"], str)
        or str(uuid.UUID(document["run_id"])) != document["run_id"]
    ):
        raise ValueError("Invalid selected preview run")
    for key, length in (("request_digest", 64), ("session_id", 32)):
        if not isinstance(document[key], str) or not re.fullmatch(
            f"[0-9a-f]{{{length}}}", document[key]
        ):
            raise ValueError("Invalid selected preview binding")
    if document["channel"] not in CHANNELS.values() or not all(
        _positive(document[key], 2**53 - 1) for key in ("generation", "sequence")
    ):
        raise ValueError("Invalid selected preview route identity")
    frame = document["frame"]
    if (
        not isinstance(frame, dict)
        or set(frame) != {"source_frame_id", "pts_ns"}
        or any(
            value is not None and (type(value) is not int or not 0 <= value <= 2**53 - 1)
            for value in frame.values()
        )
    ):
        raise ValueError("Invalid selected preview source timing")
    for key, maximum in (("source", MAX_SOURCE_SIDE), ("image", MAX_IMAGE_SIDE)):
        value = document[key]
        fields = (
            {"width", "height"}
            if key == "source"
            else {"width", "height", "path", "size", "sha256"}
        )
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or not all(_positive(value[side], maximum) for side in ("width", "height"))
        ):
            raise ValueError("Invalid selected preview dimensions")
    image = document["image"]
    if image["width"] * image["height"] > MAX_IMAGE_PIXELS:
        raise ValueError("Selected preview exceeds the desktop pixel bound")
    filename = (
        f"preview-web-{document['session_id']}-{document['generation']}-{document['sequence']}.jpg"
    )
    if (
        image["path"] != filename
        or not _positive(image["size"], MAX_IMAGE_BYTES)
        or not isinstance(image["sha256"], str)
        or not re.fullmatch("[0-9a-f]{64}", image["sha256"])
    ):
        raise ValueError("Invalid selected preview JPEG binding")
    overlays = document["overlays"]
    if (
        not isinstance(overlays, dict)
        or set(overlays) != {"players", "play", "rink"}
        or any(type(value) is not bool for value in overlays.values())
    ):
        raise ValueError("Invalid selected preview overlays")


def read_snapshot(session: Path, *, expected: dict) -> tuple[dict, bytes]:
    """Pin before decoding; a retired/replaced image is an unpresented retry."""
    owner_expected = {key: value for key, value in expected.items() if key != "sequence"}
    owner = read_owner(session, expected=owner_expected)
    if owner["status"] != "selected" or owner["channel"] == "none":
        raise ValueError("Selected preview owner has no active image route")
    manifest = PinnedFile(session, MANIFEST)
    try:
        document = json.loads(manifest.read(MAX_MANIFEST_BYTES))
        validate_manifest(document)
        if any(
            document[key] != owner[key]
            for key in ("run_id", "request_digest", "session_id", "channel", "generation")
        ):
            raise ValueError("Selected preview does not belong to the applied owner")
        if any(
            type(document.get(key)) is not type(value) or document.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("Selected preview belongs to a different route or run")
        image = PinnedFile(session, document["image"]["path"])
        try:
            if int(image.identity["size"]) != document["image"]["size"]:
                raise ValueError("Selected preview JPEG size changed")
            encoded = image.read(MAX_IMAGE_BYTES)
        finally:
            image.close()
        if hashlib.sha256(encoded).hexdigest() != document["image"]["sha256"]:
            raise ValueError("Selected preview JPEG content changed")
        manifest.check()
        if read_owner(session, expected=owner_expected) != owner:
            raise ValueError("Selected preview owner changed during the snapshot read")
        return document, encoded
    finally:
        manifest.close()


class WebPreviewPublisher:
    """Write-once JPEGs; the atomic manifest is the publication point."""

    def __init__(self, session: Path, session_id: str):
        root, document = worker_binding()
        if session != root / "work/hockeymon-session":
            raise ValueError("Selected preview escaped its private session")
        self.session = session
        self.session_id = session_id
        self.binding = {key: document[key] for key in ("run_id", "request_digest")}
        self.initial_channel = document["launch"]["preview"]
        self.sources = document["sources"]
        self.sequence = 0
        self._retained: list[Path] = []
        self.invalidate()

    def invalidate(self) -> None:
        # Retire the publication point before any referenced bytes disappear.
        (self.session / MANIFEST).unlink(missing_ok=True)
        for path in self.session.glob("preview-web-*.jpg"):
            path.unlink()
        self._retained.clear()

    def select(self, name: str, generation: int) -> None:
        self.invalidate()
        _write_ephemeral(
            self.session / OWNER,
            {
                "protocol": "hstream-web-preview-owner",
                "version": 1,
                **self.binding,
                "session_id": self.session_id,
                "channel": "none" if name == "none" else CHANNELS[name],
                "generation": generation,
                "status": "selected",
            },
        )

    def retire(self) -> None:
        retire_owner(self.session, expected={**self.binding, "session_id": self.session_id})
        self.invalidate()

    def identity(self, name: str, generation: int, metadata: dict | None, overlays: tuple) -> dict:
        self.sequence += 1
        # pts_ns elsewhere in the current Python pipeline is frame/fps-derived.
        # Only an explicitly propagated decoded_pts_ns can claim source PTS.
        metadata = metadata or {}
        return {
            "protocol": PROTOCOL,
            "version": 1,
            **self.binding,
            "session_id": self.session_id,
            "channel": CHANNELS[name],
            "generation": generation,
            "sequence": self.sequence,
            "frame": {
                "source_frame_id": metadata.get("source_frame_id"),
                "pts_ns": metadata.get("decoded_pts_ns"),
            },
            "overlays": dict(zip(("players", "play", "rink"), overlays, strict=True)),
        }

    def bind_frame_metadata(self, name: str, metadata: dict) -> dict:
        """Freeze lossless source timing on the queued job; wire v1 stays unchanged."""
        from hmlib.video.frame_timing import timestamp_from_document, timing_ns

        metadata = dict(metadata)
        timing = metadata.get("decoded_timing")
        if timing is None:
            return metadata
        timestamp_from_document(timing)
        role = metadata.get("decoded_source_role")
        allowed_role = (
            "source"
            if set(self.sources) == {"source"}
            else "right" if name == "Camera 2" else "left"
        )
        if role != allowed_role or role not in self.sources:
            raise ValueError("Decoded timing does not own the selected preview source")
        chapter = timing["chapter_index"]
        if not 0 <= chapter < len(self.sources[role]):
            raise ValueError("Decoded timing names an unadmitted chapter")
        metadata["decoded_pts_ns"] = timing_ns(timing)
        metadata["decoded_timing"] = dict(timing)
        metadata["decoded_source"] = {
            "role": role,
            "chapter_index": chapter,
            "content": dict(self.sources[role][chapter]["content"]),
            "pts": timing["pts"],
            "time_base_num": timing["time_base_num"],
            "time_base_den": timing["time_base_den"],
        }
        return metadata

    def publish(
        self, identity: dict, encoded: bytes, source: tuple[int, int], image: tuple[int, int]
    ) -> None:
        filename = (
            f"preview-web-{self.session_id}-{identity['generation']}-{identity['sequence']}.jpg"
        )
        document = {
            **identity,
            "source": dict(zip(("width", "height"), source, strict=True)),
            "image": {
                "path": filename,
                "width": image[0],
                "height": image[1],
                "size": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
        }
        validate_manifest(document)
        data = canonical_bytes(document)
        if len(data) > MAX_MANIFEST_BYTES:
            raise ValueError("Selected preview manifest exceeds its byte limit")
        # Keep the current manifest's JPEG until its successor is published.
        # Only an already superseded snapshot may be removed before writing.
        while len(self._retained) >= 2:
            self._retained[0].unlink(missing_ok=True)
            self._retained.pop(0)
        path = self.session / filename
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
        committed = False
        temporary = None
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
            with tempfile.NamedTemporaryFile(
                dir=self.session, prefix=".web-preview-", delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(data)
            os.replace(temporary, self.session / MANIFEST)
            committed = True
            self._retained.append(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if not committed:
                path.unlink(missing_ok=True)
