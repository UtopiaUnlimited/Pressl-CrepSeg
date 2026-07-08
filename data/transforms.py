from __future__ import annotations

from collections.abc import Callable


class Compose:
    def __init__(self, transforms: list[Callable[[dict], dict]]) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Identity:
    def __call__(self, sample: dict) -> dict:
        return sample
