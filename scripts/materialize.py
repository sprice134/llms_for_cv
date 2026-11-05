#!/usr/bin/env python3
"""Expand the archived task manifests and scored model outputs in place."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for folder in ('results/manifests', 'results/evaluation'):
        for source in sorted((ROOT / folder).glob('*.gz')):
            destination = source.with_suffix('')
            if destination.exists():
                continue
            with gzip.open(source, 'rb') as inp, destination.open('wb') as out:
                shutil.copyfileobj(inp, out)
            print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
