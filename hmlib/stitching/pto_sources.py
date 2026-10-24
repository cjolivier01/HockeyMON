"""Lightweight PTO source-path rebinding for calibration publication."""

import json
import re
import shlex
from collections.abc import Sequence
from pathlib import Path


def rewrite_pto_sources(
    project: Path,
    *,
    images: Sequence[str] | None = None,
    source_directory: Path | None = None,
    target_directory: Path | None = None,
) -> None:
    """Rebind quoted image tokens without matching their escaped text as paths."""
    lines = project.read_text(encoding="utf-8").splitlines()
    image_index = 0
    for index, line in enumerate(lines):
        if not line.lstrip().startswith(("i ", "i\t")):
            continue
        tokens = list(re.finditer(r'(?<!\S)n"(?:\\.|[^"\\])*"', line))
        if len(tokens) != 1:
            raise ValueError("Cached PTO does not contain one filename per input image")
        token = tokens[0]
        if images is not None:
            if image_index >= len(images):
                raise ValueError("Cached PTO contains more than two input images")
            filename = images[image_index]
        else:
            filename = shlex.split(token.group())[0][1:]
            path = Path(filename)
            if not path.is_absolute():
                path = project.parent / path
            if path.is_relative_to(source_directory):
                filename = str(target_directory / path.relative_to(source_directory))
        lines[index] = (
            line[: token.start()]
            + "n"
            + json.dumps(filename, ensure_ascii=False)
            + line[token.end() :]
        )
        image_index += 1
    project.write_text("\n".join(lines) + "\n", encoding="utf-8")
