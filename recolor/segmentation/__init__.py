"""Segmentation stage: SAM 2.1 proposals -> clean region partition -> color groups.

Public entry points (see docs/ARCHITECTURE.md §3.2):

- `SamMasker().generate(image, detail)` — overlapping mask proposals (GPU, lazy model).
- `build_regions(image, albedo, masks, detail)` — every pixel labelled 0..N-1.
- `group_regions(labels, albedo, info, max_groups, delta_e)` — regions clustered by
  median albedo into `ColorGroup`s plus a group map; `regroup`, `merge_groups`,
  `split_group`, `move_regions` edit the grouping while keeping region ids stable.

Importing this package does not import sam2 or load any model.
"""
from __future__ import annotations

from .grouping import group_regions, merge_groups, move_regions, regroup, split_group
from .hierarchy import REGION_PRESETS, build_regions
from .sam_masks import DETAIL_PRESETS, SamMasker, is_loaded, warmup
from .superpixels import slic_labels

__all__ = [
    "DETAIL_PRESETS", "REGION_PRESETS", "SamMasker", "warmup", "is_loaded",
    "slic_labels", "build_regions",
    "group_regions", "regroup", "merge_groups", "split_group", "move_regions",
]
