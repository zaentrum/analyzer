from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from captions_fake import FFPROBE_715, FakeProbe, Video

from analyzer.pipelines import captions


@pytest.fixture
def fake_ffprobe(tmp_path: Path) -> Iterator[Callable[..., FakeProbe]]:
    """Make a made-up ffprobe saying version (the pipeline's 7.1.5 by
    default) and answering of video (a ten-minute one by default), each in a
    folder of its own."""
    made = 0

    def make(version: str = FFPROBE_715, video: Video | None = None) -> FakeProbe:
        nonlocal made
        made += 1
        return FakeProbe(tmp_path / f"ffprobe-{made}", version, video)

    captions.probe.cache_clear()
    yield make
    captions.probe.cache_clear()
