"""Bounded original-video optics import, matching native HStream's camera policy.

Only stationary, representable radial prescriptions are imported. Unsupported
recording modes retain their selected profile and produce an explicit warning.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from hmlib.stitching.imu_synchronization import _insta_record
from hmlib.stitching.input_lens import InputLens, read_input_lens

logger = logging.getLogger(__name__)
_LIMIT = 4 * 1024 * 1024


class UnsupportedLens(ValueError):
    """The recording is valid but has no supported static lens prescription."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(data):
    value = data.split(b"\0", 1)[0]
    _require(len(value) <= 4096, "Lens metadata string exceeds limit")
    return value.decode("utf-8")


def _protobuf(data):
    _require(len(data) <= _LIMIT, "Lens metadata exceeds size limit")
    fields, position = {}, 0

    def varint():
        nonlocal position
        value = 0
        for shift in range(0, 64, 7):
            _require(position < len(data), "Truncated protobuf integer")
            byte = data[position]
            position += 1
            _require(shift != 63 or byte <= 1, "Invalid protobuf integer")
            value |= (byte & 127) << shift
            if not byte & 128:
                return value
        raise ValueError("Invalid protobuf integer")

    while position < len(data):
        key = varint()
        number, kind = key >> 3, key & 7
        _require(
            0 < number <= 0x1FFFFFFF and number not in fields and len(fields) < 4096,
            "Invalid/repeated lens metadata field",
        )
        if kind == 0:
            value = varint()
        else:
            _require(kind in (1, 2, 5), "Unsupported protobuf wire type")
            length = varint() if kind == 2 else 8 if kind == 1 else 4
            _require(length <= len(data) - position, "Truncated protobuf field")
            value = data[position : position + length]
            position += length
        fields[number] = kind, value
    return fields


def _field(fields, number, kind, default):
    if number not in fields:
        return default
    actual, value = fields[number]
    _require(actual == kind, "Invalid lens metadata field type")
    return value


def _child(fields, number):
    return _protobuf(_field(fields, number, 2, b""))


def _scalar(fields, number):
    return _field(fields, number, 0, 0)


def _string(fields, number):
    return _text(_field(fields, number, 2, b""))


@dataclass
class VideoLens:
    width: int
    height: int
    model: str = ""
    mode: str = ""
    stabilization: str = ""
    recording_mode: dict[str, str] = field(default_factory=dict)
    horizontal_fov: float | None = None
    vertical_fov: float | None = None
    calibration_fov: float | None = None
    lens: InputLens | None = None
    unavailable: str = ""


def _unavailable(result, reason):
    result.unavailable = f"{result.model}: {reason}"
    return result


def _fit(width, height, angle, extent=0):
    _require(16 <= width <= 32768 and 16 <= height <= 32768, "Invalid lens dimensions")
    diagonal = max(extent, math.hypot(width, height) / 2)
    radius = min(width, height) / 2
    radii = np.linspace(0, diagonal, 20001)
    theta = angle(radii)
    _require(
        np.isfinite(theta).all()
        and np.all(np.diff(theta) > 0)
        and abs(theta[0]) < 1e-10
        and 0 < theta[-1] < math.pi,
        "Lens mapping is not centered and monotonic",
    )
    x = theta / theta[-1]
    design = x[:, None] ** np.arange(1, 5)
    q = np.linalg.lstsq(design, radii / radius, rcond=None)[0]
    fitted = design @ q
    _require(np.all(np.diff(fitted) > 0), "Fitted lens folds at image edge")
    if np.max(np.abs(fitted * radius - radii)) > max(width, height) / 7680:
        raise UnsupportedLens("Lens approximation exceeds one pixel at 8K")
    _require(fitted[-1] >= 1, "Lens fit does not cover image")
    lo, hi = 0.0, 1.0
    for _ in range(64):
        middle = (lo + hi) / 2
        if np.dot(middle ** np.arange(1, 5), q) < 1:
            lo = middle
        else:
            hi = middle
    scale = (lo + hi) / 2
    fov = math.degrees(width * scale * theta[-1] / radius)
    vertical = math.degrees(2 * float(angle(height / 2)))
    lens = read_input_lens(
        {
            "projection": "full-frame-fisheye",
            "optimization": "fixed",
            "distortion": [q[3] * scale**4, q[2] * scale**3, q[1] * scale**2],
        }
    )
    _require(0 < fov < 360 and 0 < vertical <= 180, "Unsupported lens field of view")
    return fov, lens


def _inverse(radius, maximum):
    angles = np.arange(30000) * math.pi / 30000
    values = radius(angles)
    indices = np.flatnonzero(values >= maximum)
    _require(len(indices) > 0, "Source lens does not cover image")
    end = indices[0]
    _require(
        np.isfinite(values[: end + 1]).all() and np.all(np.diff(values[: end + 1]) > 0),
        "Nonmonotonic source lens model",
    )
    upper = angles[end]

    def angle(r):
        r = np.asarray(r)
        lo, hi = np.zeros_like(r, dtype=float), np.full_like(r, upper, dtype=float)
        for _ in range(48):
            mid = (lo + hi) / 2
            lower = radius(mid) < r
            lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
        return np.where(r == 0, 0, (lo + hi) / 2)

    return angle


def _gpmf(data):
    _require(len(data) <= _LIMIT, "GoPro metadata exceeds size limit")
    position = 0
    while position < len(data):
        if data[position] == 0:
            _require(not any(data[position:]), "Nonzero bytes after GPMF padding")
            break
        _require(len(data) - position >= 8, "Truncated GPMF header")
        key, kind, width, count = struct.unpack_from(">4sBBH", data, position)
        length = width * count
        padded = (length + 3) & ~3
        _require(padded <= len(data) - position - 8, "Truncated GPMF payload")
        yield key.decode("ascii"), kind, data[position + 8 : position + 8 + length]
        position += 8 + padded


def parse_gopro_lens(data: bytes) -> VideoLens:
    global_tags, lens_tags = {}, {}
    for key, kind, payload in _gpmf(data):
        if key != "DEVC" or kind:
            continue
        tags = {}
        for name, tag_kind, value in _gpmf(payload):
            _require(name not in tags and len(tags) < 4096, "Duplicate/excessive GoPro tags")
            tags[name] = tag_kind, value
        if _text(tags.get("DVID", (0, b""))[1]) == "FOVL":
            _require(not lens_tags, "Multiple GoPro FOVL models")
            lens_tags = tags
        elif "MINF" in tags:
            _require(not global_tags, "Multiple GoPro settings blocks")
            global_tags = tags

    def text(tags, key):
        return _text(tags.get(key, (0, b""))[1])

    def floats(tags, key):
        kind, payload = tags.get(key, (0, b""))
        _require(kind == ord("f") and payload and len(payload) % 4 == 0, f"Invalid GoPro {key}")
        value = np.frombuffer(payload, ">f4").astype(float)
        _require(np.isfinite(value).all(), "Nonfinite GoPro coefficient")
        return value

    def number(tags, key):
        value = floats(tags, key)
        _require(len(value) == 1, "Invalid GoPro scalar")
        return float(value[0])

    kind, dimensions = global_tags.get("VRES", (0, b""))
    _require(kind == ord("L") and len(dimensions) == 8, "Missing GoPro dimensions")
    w, h = struct.unpack(">II", dimensions)
    _require(w > 0 and h > 0, "Invalid GoPro dimensions")
    result = VideoLens(
        w, h, text(global_tags, "MINF"), text(lens_tags, "VFOV"), text(global_tags, "EISA")
    )
    result.recording_mode = {
        "vendor": "gopro",
        "model": result.model,
        "width": str(w),
        "height": str(h),
        "digital_lens": result.mode,
        "stabilization": result.stabilization,
        **{
            name: text(global_tags, tag)
            for name, tag in (("horizon", "HCTL"), ("projection", "PRJT"), ("firmware", "FMWR"))
        },
    }
    if "LPID" in global_tags:
        result.recording_mode["lens_id"] = text(global_tags, "LPID")
    fps_kind, fps = global_tags.get("VFPS", (0, b""))
    if fps_kind == ord("L") and len(fps) == 8:
        n, d = struct.unpack(">II", fps)
        if n and d:
            result.recording_mode["frame_rate"] = f"{n/d:.3f}"
    if "ZOOM" in global_tags:
        result.recording_mode["zoom"] = format(number(global_tags, "ZOOM"), ".9g")
    if not lens_tags:
        return _unavailable(result, "no FOVL prescription or axis FOV")
    if text(global_tags, "PRJT") != "GPRO" or result.mode != "W":
        return _unavailable(result, "only GoPro Wide radial metadata is supported")
    if "boost" in result.stabilization.lower() or (
        "ABSC" in lens_tags and number(lens_tags, "ABSC") != 0
    ):
        return _unavailable(result, "AutoBoost requires a changing lens model")
    if any(not np.array_equal(floats(lens_tags, key), [1]) for key in ("MAPX", "MAPY")):
        return _unavailable(result, "digital lens warp is not radial")
    result.recording_mode["warp"] = "identity"
    _require(
        all(abs(number(lens_tags, key) - w / h) < 1e-5 for key in ("ARUW", "ARWA")),
        "GoPro aspect ratio differs from recorded image",
    )
    for key, attribute, maximum in (
        ("XFOV", "horizontal_fov", 360),
        ("YFOV", "vertical_fov", 180.000001),
    ):
        if key in lens_tags:
            value = number(lens_tags, key)
            _require(0 < value < maximum, "Invalid GoPro axis FOV")
            setattr(result, attribute, value)
    if any(key not in lens_tags for key in ("POLY", "PYCF", "ZMPL")):
        return _unavailable(result, "recording omits radial lens prescription")
    poly = floats(lens_tags, "POLY")
    basis = lens_tags["PYCF"][1]
    _require(
        len(poly) == 8
        and len(basis) == 64
        and all(basis[8 * i : 8 * i + 3] == f"r{i}\0".encode() for i in range(8)),
        "Unsupported GoPro polynomial basis",
    )
    zoom, diagonal = number(lens_tags, "ZMPL"), math.hypot(w, h) / 2
    _require(0 < zoom < 10, "Invalid GoPro zoom normalization")

    def angle(r):
        return np.polynomial.polynomial.polyval(np.asarray(r) * zoom / diagonal, poly)

    for key, radius in (("XFOV", w / 2), ("YFOV", h / 2), ("ZFOV", diagonal)):
        if key in lens_tags:
            _require(
                abs(number(lens_tags, key) - math.degrees(2 * float(angle(radius)))) < 0.25,
                "GoPro prescription disagrees with axis FOV",
            )
    if result.horizontal_fov is None:
        result.horizontal_fov = math.degrees(2 * float(angle(w / 2)))
    if result.vertical_fov is None:
        result.vertical_fov = math.degrees(2 * float(angle(h / 2)))
    _require(
        0 < result.horizontal_fov < 360 and 0 < result.vertical_fov <= 180,
        "Invalid derived GoPro FOV",
    )
    try:
        result.calibration_fov, result.lens = _fit(w, h, angle)
    except UnsupportedLens as error:
        return _unavailable(result, str(error))
    return result


def parse_insta_lens(data: bytes) -> VideoLens:
    fields = _protobuf(data)
    dimensions = _child(fields, 19)
    w, h = _scalar(dimensions, 1), _scalar(dimensions, 2)
    result = VideoLens(
        w,
        h,
        _string(fields, 2),
        f"FOV {_scalar(fields, 47)}",
        "FlowState" if _scalar(fields, 42) else "Off",
    )
    _require(bool(result.model), "Insta360 camera model is missing")
    if _scalar(fields, 42) or _scalar(fields, 43):
        return _unavailable(result, "FlowState/dewarping is baked into output")
    if _string(fields, 111):
        return _unavailable(result, "v6 lens model is unsupported")
    tokens = _string(fields, 54).split("_")
    if len(tokens) != 21:
        return _unavailable(result, "no supported single-lens v3 calibration")
    v = list(map(float, tokens))
    _require(all(math.isfinite(value) for value in v), "Invalid Insta360 coefficient")
    if v[0] != 1 or v[17:19] != [w, h] or abs(v[2] - v[3]) > 1e-5 or v[15] or v[16]:
        return _unavailable(result, "cropped, asymmetric or tangential model")
    crop = _child(fields, 27)
    if crop and [_scalar(crop, i) for i in range(1, 5)] != [w, h, w, h]:
        return _unavailable(result, "unsupported recording crop")

    def radius(theta):
        r = np.sin(theta) / (np.cos(theta) + v[1])
        r2 = r * r
        return v[2] * r * (1 + r2 * (v[12] + r2 * (v[13] + r2 * v[14])))

    extent = math.hypot(max(v[4], w - v[4]), max(v[5], h - v[5]))
    angle = _inverse(radius, extent)
    result.horizontal_fov = math.degrees(float(angle(v[4]) + angle(w - v[4])))
    result.vertical_fov = math.degrees(float(angle(v[5]) + angle(h - v[5])))
    try:
        result.calibration_fov, result.lens = _fit(w, h, angle, extent)
    except UnsupportedLens as error:
        return _unavailable(result, str(error))
    offset = ((v[4] - w / 2) / w, (v[5] - h / 2) / h)
    if any(abs(value) > 0.01 for value in offset):
        result.lens = None
        return _unavailable(result, "optical center exceeds supported fit margin")
    result.lens = replace(result.lens, center_offset=offset)
    return result


def parse_dji_lens(data: bytes, width: int, height: int) -> VideoLens:
    clip = _child(_protobuf(data), 1)
    header = _child(clip, 1)
    result = VideoLens(width, height, _string(header, 10))
    if _string(header, 1) not in ("dvtm_ac204.proto", "dvtm_wm169.proto"):
        raise UnsupportedLens("Unsupported DJI metadata schema")
    if 9 not in clip or _scalar(_child(clip, 9), 1):
        return _unavailable(result, "stabilization is enabled or unknown")
    focal, distortion = _child(clip, 8), _child(clip, 3)
    if 1 not in focal or 1 not in distortion:
        return _unavailable(result, "recording omits focal length/distortion")
    fx_bytes, coefficients = _field(focal, 1, 5, b""), _field(distortion, 1, 2, b"")
    _require(len(fx_bytes) == 4 and len(coefficients) == 16, "Invalid DJI coefficient layout")
    (fx,) = struct.unpack("<f", fx_bytes)
    k = struct.unpack("<4f", coefficients)
    _require(
        math.isfinite(fx) and fx > 0 and all(map(math.isfinite, k)), "Invalid DJI coefficients"
    )

    def radius(theta):
        t2 = theta * theta
        return fx * theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))

    angle = _inverse(radius, math.hypot(width, height) / 2)
    result.horizontal_fov, result.vertical_fov = (
        math.degrees(2 * float(angle(size / 2))) for size in (width, height)
    )
    try:
        result.calibration_fov, result.lens = _fit(width, height, angle)
    except UnsupportedLens as error:
        return _unavailable(result, str(error))
    result.mode, result.stabilization = "OpenCV fisheye", "Off"
    return result


class _Mp4:
    def __init__(self, stream):
        self.stream = stream
        self.size = stream.seek(0, 2)
        self.bytes_read = self.boxes = 0

    def read(self, position, count):
        _require(
            0 <= count <= _LIMIT
            and self.bytes_read + count <= 8 * _LIMIT
            and 0 <= position <= self.size
            and count <= self.size - position,
            "MP4 metadata read limit exceeded",
        )
        self.bytes_read += count
        self.stream.seek(position)
        data = self.stream.read(count)
        _require(len(data) == count, "Truncated MP4 metadata")
        return data

    def atoms(self, position, length, stop=None):
        end = position + length
        _require(end <= self.size, "MP4 container outside file")
        while position < end:
            self.boxes += 1
            _require(self.boxes <= 4096 and end - position >= 8, "Invalid MP4 atom count/length")
            size, kind = struct.unpack(">I4s", self.read(position, 8))
            header = 8
            if size == 1:
                _require(end - position >= 16, "Truncated extended atom")
                size, header = int.from_bytes(self.read(position + 8, 8), "big"), 16
            elif size == 0:
                size = end - position
            _require(header <= size <= end - position, "MP4 atom outside container")
            yield kind, position + header, size - header
            position += size
            if kind == stop:
                break

    def track(self, position, length, depth=0):
        _require(depth < 5, "MP4 nesting limit exceeded")
        result = {}
        for kind, offset, size in self.atoms(position, length):
            if kind in (b"mdia", b"minf", b"stbl"):
                result.update(self.track(offset, size, depth + 1))
            else:
                result.setdefault(kind, (offset, size))
        return result


@lru_cache(maxsize=256)
def _read_cached(path, identity):
    video = Path(path)
    insta = _insta_record(video, 1, _LIMIT)
    if insta:
        _require(insta[1] <= _LIMIT, "Insta360 lens metadata exceeds size limit")
    vendor = parse_insta_lens(insta[0]) if insta else None
    with video.open("rb") as stream:
        mp4 = _Mp4(stream)
        width = height = 0
        dji = {}
        for kind, top_offset, top_size in mp4.atoms(0, mp4.size, stop=b"moov"):
            if kind != b"moov":
                continue
            for kind, offset, size in mp4.atoms(top_offset, top_size):
                if kind == b"udta":
                    for tag, at, length in mp4.atoms(offset, size):
                        if tag == b"GPMF":
                            _require(vendor is None, "Multiple camera lens metadata blocks")
                            vendor = parse_gopro_lens(mp4.read(at, length))
                if kind != b"trak":
                    continue
                leaves = mp4.track(offset, size)
                if b"stsd" not in leaves or leaves[b"stsd"][1] < 16:
                    continue
                at, length = leaves[b"stsd"]
                description = mp4.read(at, min(length, 100))
                codec = description[12:16]
                if codec == b"djmd":
                    dji = leaves
                if codec in (b"hvc1", b"hev1", b"avc1"):
                    _require(
                        len(description) >= 44 and width == 0,
                        "Invalid or multiple video sample descriptions",
                    )
                    width, height = struct.unpack(">HH", description[40:44])
        if vendor is not None:
            _require(
                width > 0 and (width, height) == (vendor.width, vendor.height),
                "Lens dimensions differ from video",
            )
            result = vendor
        elif dji:
            offset_kind = b"co64" if b"co64" in dji else b"stco"
            _require(offset_kind in dji and b"stsz" in dji, "DJI metadata lacks sample table")
            at, length = dji[offset_kind]
            n = 8 if offset_kind == b"co64" else 4
            _require(length >= 8 + n and dji[b"stsz"][1] >= 12, "Truncated DJI sample table")
            offsets = mp4.read(at, 8 + n)
            sizes = mp4.read(dji[b"stsz"][0], min(dji[b"stsz"][1], 16))
            _require(
                int.from_bytes(offsets[4:8], "big") > 0 and int.from_bytes(sizes[8:12], "big") > 0,
                "Empty DJI sample table",
            )
            size = int.from_bytes(sizes[4:8], "big")
            if not size:
                _require(len(sizes) >= 16, "Missing DJI sample size")
                size = int.from_bytes(sizes[12:16], "big")
            result = parse_dji_lens(
                mp4.read(int.from_bytes(offsets[8 : 8 + n], "big"), size), width, height
            )
        else:
            raise UnsupportedLens(
                "No supported lens metadata; original camera files may be required"
            )
    _require(_identity(video) == identity, "Source video changed while reading lens metadata")
    return result


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_video_lens(path: str | Path) -> VideoLens:
    video = Path(path).resolve(strict=True)
    return _read_cached(str(video), _identity(video))


def hugin_fov(
    coverage: float, width: int, height: int, lens: InputLens, *, to_physical=False
) -> float:
    """Convert physical coverage to/from the selected model's radial normalization."""
    _require(
        width >= 16 and height >= 16 and 0 < coverage < (360 if to_physical else 180),
        "Cannot combine FOV with selected lens model",
    )
    radius = min(width, height) / 2
    a, b, c = lens.distortion

    def radial(t):
        return t * (1 - a - b - c + t * (c + t * (b + t * a)))

    ox, oy = lens.center_offset[0] * width, lens.center_offset[1] * height
    edge = math.hypot(width / 2 + abs(ox), height / 2 + abs(oy)) / radius
    samples = np.arange(10001) * 0.001
    values = radial(samples)
    reached = np.flatnonzero(values >= edge)
    _require(len(reached) > 0, "Selected lens does not cover image")
    end = reached[0]
    _require(
        np.isfinite(values[: end + 1]).all() and np.all(np.diff(values[: end + 1]) > 0),
        "Selected lens folds before edge",
    )
    upper = samples[end]

    def derivative(t):
        return 1 - a - b - c + t * (2 * c + t * (3 * b + t * 4 * a))

    extrema = [0, upper]
    if a == 0:
        if b:
            extrema.append(-c / (3 * b))
    elif 36 * b * b - 96 * a * c >= 0:
        root = math.sqrt(36 * b * b - 96 * a * c)
        extrema.extend(((-6 * b - root) / (24 * a), (-6 * b + root) / (24 * a)))
    _require(
        all(derivative(t) > 0 for t in extrema if 0 <= t <= upper),
        "Selected lens has a folding radial branch",
    )

    def inverse(r):
        lo, hi = 0.0, upper
        for _ in range(56):
            mid = (lo + hi) / 2
            if radial(mid) < r:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lx, rx = -width / 2 - ox, width / 2 - ox
    lr, rr = math.hypot(lx, oy), math.hypot(rx, oy)
    lt, rt = inverse(lr / radius), inverse(rr / radius)

    def angle(fov):
        def ray(x, r, t):
            if r == 0:
                return np.array([0.0, 0.0, 1.0])
            theta = (
                math.atan(t * 2 * radius / width * math.tan(fov / 2))
                if lens.projection == "rectilinear"
                else t * fov * radius / width
            )
            return np.array([math.sin(theta) * x / r, -math.sin(theta) * oy / r, math.cos(theta)])

        return math.acos(float(np.clip(ray(lx, lr, lt) @ ray(rx, rr, rt), -1, 1)))

    if to_physical:
        return math.degrees(angle(math.radians(coverage)))
    target, lo, hi, last = math.radians(coverage), 0.0, 0.0, -1.0
    limit = (math.pi if lens.projection == "rectilinear" else 2 * math.pi) - 1e-6
    for i in range(1, 2049):
        hi = limit * i / 2048
        value = angle(hi)
        _require(math.isfinite(value) and value > last, "FOV outside selected lens coverage")
        last = value
        if value >= target:
            break
    _require(last >= target, "FOV outside selected lens coverage")
    for _ in range(56):
        mid = (lo + hi) / 2
        if angle(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.degrees((lo + hi) / 2)


def _matching_profile(recording, profiles):
    matches = []
    required = {
        "vendor",
        "model",
        "width",
        "height",
        "frame_rate",
        "digital_lens",
        "stabilization",
        "horizon",
        "zoom",
        "projection",
        "warp",
    }
    for name, profile in profiles.items():
        constraints = profile.get("metadata_match")
        if not constraints:
            continue
        _require(
            isinstance(constraints, dict) and required <= constraints.keys(),
            f"Incomplete camera metadata_match: {name}",
        )
        if "lens_id" in recording and "lens_id" not in constraints:
            continue
        if all(
            recording.get(key)
            in [str(item) for item in (value if isinstance(value, list) else [value])]
            for key, value in constraints.items()
        ):
            matches.append(profile)
    return matches[0] if len(matches) == 1 else None


def resolve_video_lens(config, settings, videos):
    """Resolve all selected chapters together, returning settings and provenance notes.

    Independent manual components stay unchanged. Automatic failures retain the
    selected profile's defaults and are surfaced to callers and logs.
    """
    stitch = (config or {}).get("stitching") or {}
    automatic_fov, automatic_lens = settings.camera_auto_fov, settings.camera_auto_lens
    if not automatic_fov and not automatic_lens:
        return settings, ["Camera optics: manual / selected profile"]
    if stitch.get("auto_input_lens_frozen") is True:
        return settings, ["Camera optics: saved experiment snapshot"]
    if not 0 < len(videos) <= 256:
        raise ValueError("Automatic lens import requires between one and 256 selected chapters")
    profiles = stitch.get("camera_configs")
    if profiles is None:
        import yaml

        with (Path(__file__).resolve().parents[1] / "config" / "baseline.yaml").open() as stream:
            profiles = yaml.safe_load(stream)["stitching"]["camera_configs"]
    preset = profiles.get(settings.camera_config)
    resolved = settings
    if preset:
        changes = {}
        if automatic_fov:
            changes.update(
                horizontal_fov=float(preset["horizontal_fov"]),
                vertical_fov=float(preset["vertical_fov"]),
            )
        if automatic_lens:
            changes["input_lens"] = read_input_lens(preset.get("input_lens"))
        resolved = replace(resolved, **changes)
    values, notes = [], []
    for video in videos:
        try:
            values.append(read_video_lens(video))
        except (OSError, ValueError, struct.error) as error:
            notes.append(
                f"Auto camera unavailable for {Path(video).name}: {error}; using {settings.camera_config} defaults"
            )
    if notes:
        for note in notes:
            logger.warning("%s", note)
        return resolved, notes
    first = values[0]
    if any(value != first for value in values[1:]):
        note = "Camera/chapter metadata differs; automatic optics retain shared profile defaults"
        logger.warning("%s", note)
        return resolved, [note]
    notes.append(
        f"Original recording: {first.model}; {first.width}×{first.height}; {first.mode}; {first.stabilization}"
    )
    profile = _matching_profile(first.recording_mode, profiles)
    lens = resolved.input_lens
    if automatic_lens:
        if first.lens is not None:
            lens = first.lens
            notes.append("Lens correction imported from original video")
        elif profile:
            lens = read_input_lens(profile.get("input_lens"))
            notes.append(f"Lens correction from verified mode profile: {profile['display_name']}")
        else:
            notes.append(
                f"Lens correction retains profile defaults: {first.unavailable or 'no supported prescription'}"
            )
    horizontal, vertical = resolved.horizontal_fov, resolved.vertical_fov
    if automatic_fov:
        if profile:
            profile_lens = read_input_lens(profile.get("input_lens"))
            try:
                horizontal = (
                    float(profile["horizontal_fov"])
                    if lens == profile_lens
                    else hugin_fov(
                        hugin_fov(
                            float(profile["horizontal_fov"]),
                            first.width,
                            first.height,
                            profile_lens,
                            to_physical=True,
                        ),
                        first.width,
                        first.height,
                        lens,
                    )
                )
                vertical = float(profile["vertical_fov"])
                notes.append("Camera FOV from verified recording-mode profile")
            except ValueError as error:
                notes.append(f"Mode FOV cannot be combined with selected lens: {error}")
        if first.horizontal_fov is not None:
            try:
                horizontal = (
                    first.calibration_fov
                    if first.lens is not None and automatic_lens
                    else hugin_fov(first.horizontal_fov, first.width, first.height, lens)
                )
                notes.append(
                    f"Horizontal FOV imported from video: coverage {first.horizontal_fov:.3f}°, calibration {horizontal:.3f}°"
                )
            except ValueError as error:
                notes.append(f"Video FOV cannot be combined with selected lens: {error}")
        if first.vertical_fov is not None:
            vertical = first.vertical_fov
            notes.append(f"Vertical FOV imported for provenance: {vertical:.3f}°")
    if lens != InputLens() and resolved.mapping_backend != "nona":
        raise ValueError(
            "Automatic input lens correction requires mapping_backend=nona; select manual neutral optics for OpenCV"
        )
    for note in notes:
        logger.info("%s", note)
    return (
        replace(resolved, input_lens=lens, horizontal_fov=horizontal, vertical_fov=vertical),
        notes,
    )
