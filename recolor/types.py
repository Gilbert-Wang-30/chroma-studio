"""Shared records passed between pipeline stages and serialized to job.json.

Everything here is plain data. Numpy arrays never live in these records; they are
stored as files under the job directory (see docs/ARCHITECTURE.md, "Job artifacts").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

Lab = tuple[float, float, float]

#: The 'shiny' badge's cue: a group at least this share of whose pixels are glints (sensor-
#: clipped or a specular spike). `segmentation.grouping._finish` computes the badge with it,
#: and a stored badge is checked against it again when a group record is read, so a job
#: analysed when the badge followed the broader highlight share loses the stale badge on
#: load instead of keeping it until a regroup.
SHINY_GLINT_SHARE = 0.2


def _from_dict(cls, d: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Region:
    """One segment of the working-resolution image. Regions may be disconnected
    (a SAM mask covering both mirrors is one region) but never overlap."""
    id: int
    area: int                                  # pixels at working resolution
    bbox: tuple[int, int, int, int]            # x0, y0, x1, y1 (exclusive), working res
    albedo_lab: Lab                            # median albedo of the region, CIE Lab (L in 0..100)
    albedo_hex: str                            # sRGB hex of albedo_lab, e.g. "#c0392b"
    group_id: int
    touches_border: bool
    source: str                                # 'sam' | 'superpixel' | 'split' | 'prompt' | 'part' | 'small' | 'text' | 'wheel' | 'named' | 'kind'
    confidence: float = 0.0                    # SAM predicted IoU; 0 for superpixels
    backdrop: bool = False                     # the foreground matte (or the border rule) calls it backdrop
    shiny: float = 0.0                         # share of highlight pixels (clipped, glints, strong residual), 0..1
    glint: float = 0.0                         # share of glint pixels (sensor-clipped or a specular spike), 0..1
    chrome: bool = False                       # advisory: reflects its surroundings like chrome or glass
    # A detected part (smallparts.find_kind_parts): the kind's key ("shock_spring"), its display
    # names for one and for several instances, and which instance of the kind this region is.
    # '' / -1 for every other region (and for every region of a job analysed before parts).
    part_kind: str = ""
    part_label: str = ""
    part_plural: str = ""
    part_instance: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        d["albedo_lab"] = tuple(d["albedo_lab"])
        return _from_dict(cls, d)


@dataclass
class ColorGroup:
    """A set of regions that share one original paint/material color."""
    id: int
    name: str                                  # human color name, e.g. "Crimson"
    albedo_lab: Lab                            # area-weighted median albedo of the group
    albedo_hex: str
    area: int
    area_frac: float                           # of the whole working image, 0..1
    region_ids: list[int]
    hue_family: str                            # red|orange|yellow|green|cyan|blue|purple|magenta|neutral
    locked: bool = False                       # user says: never recolor this group
    is_background: bool = False                # backdrop, floor or wall (matte rule, or the border rule); user can toggle
    shiny: float = 0.0                         # area-weighted share of highlight pixels, 0..1 (advisory)
    glint: float = 0.0                         # area-weighted share of glint pixels, 0..1 (the 'shiny' badge's cue)
    finish: str = ""                           # badge: '' | 'shiny' | 'chrome'; advisory, except that the engine never
                                               # treats a 'chrome' group as white paint (engine.NEUTRAL_*: its neutral
                                               # floor is its reflections)
    # A part group: most of its area is detected-part regions of one kind (``Region.part_kind``);
    # '' for a colour group. ``part_instances`` counts the part instances it holds.
    part: str = ""
    part_label: str = ""
    part_plural: str = ""
    part_instances: int = 0
    # The Groups panel's view (``grouping.annotate_groups``): a tiny group of the object (below
    # MINOR_FRAC of the object area, neither a part nor lettering) that is its ``parent``'s colour
    # under other light (the group owning most of the ring around it: within MINOR_DE, or, both
    # coloured, of its body colour; ``grouping._lighting_variant``) is ``minor``, shown under the
    # collapsed Minor divider; a tiny group of a colour of its own is not. ``parent`` is -1 for a
    # group that is not minor. View only: a minor group is painted and edited like any other.
    minor: bool = False
    parent: int = -1
    # The albedo the engine repaints this group from when it is not its own median: the
    # instances of a part split by instance keep the part's (``grouping.split_instances``), so
    # the split alone leaves the render as it was. None for every other group; any change of
    # the group's regions (merge, move, split, regroup) drops it.
    ref_lab: Optional[Lab] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ColorGroup":
        d = dict(d)
        d["albedo_lab"] = tuple(d["albedo_lab"])
        d["region_ids"] = list(d["region_ids"])
        if d.get("ref_lab") is not None:
            d["ref_lab"] = tuple(float(v) for v in d["ref_lab"])
        if d.get("finish") == "shiny" and float(d.get("glint") or 0.0) < SHINY_GLINT_SHARE:
            d["finish"] = ""                  # a badge from before the glint rule (advisory only)
        return _from_dict(cls, d)


@dataclass
class PaletteColor:
    hex: str
    lab: Lab
    weight: float                              # 0..1; weights sum to 1 within a palette
    name: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PaletteColor":
        d = dict(d)
        d["lab"] = tuple(d["lab"])
        return _from_dict(cls, d)


@dataclass
class PaletteSource:
    """A reference image the palette was distilled from."""
    url: str                                   # page or file URL for attribution
    title: str
    license: str
    thumb: str                                 # API path of the cached thumbnail, e.g. /api/palettes/<pid>/sources/0.jpg

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PaletteSource":
        return _from_dict(cls, d)


@dataclass
class Palette:
    id: str
    prompt: str
    colors: list[PaletteColor]
    sources: list[PaletteSource]
    method: str                                # 'images' | 'theme' | 'parsed' | 'mixed' | 'fallback'
    created: float

    def to_dict(self) -> dict:
        return {
            "id": self.id, "prompt": self.prompt, "method": self.method, "created": self.created,
            "colors": [c.to_dict() for c in self.colors],
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Palette":
        return cls(
            id=d["id"], prompt=d["prompt"], method=d.get("method", "images"), created=d.get("created", 0.0),
            colors=[PaletteColor.from_dict(c) for c in d.get("colors", [])],
            sources=[PaletteSource.from_dict(s) for s in d.get("sources", [])],
        )


@dataclass
class RenderOptions:
    """Knobs for the recoloring engine. All optional; defaults give the most realistic result."""
    mode: str = "shift"                        # 'shift' keeps albedo texture (Lab offset); 'flat' paints the group flat
    texture: float = 1.0                       # blend between flat (0) and shift (1); only used when mode == 'shift'
    feather_px: float = 1.5                    # soft edge between groups, pixels at the render resolution
    keep_residual: bool = True                 # add the specular/saturation residual back
    residual_tint: float = 0.0                 # 0 = highlights keep original color, 1 = tinted toward new color
    shading_strength: float = 1.0              # 1 = original shading; <1 flattens, >1 deepens
    saturation: float = 1.0                    # multiplier on target chroma
    sharpen_edges: bool = False                # reserved

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RenderOptions":
        return _from_dict(cls, d or {})


# Mapping: group id -> hex color, or None to leave the group untouched.
Mapping = dict[int, Optional[str]]


def mapping_from_json(d: dict) -> Mapping:
    return {int(k): (v if v else None) for k, v in (d or {}).items()}


def mapping_to_json(m: Mapping) -> dict:
    return {str(k): v for k, v in m.items()}


STAGES = ["ingest", "intrinsic", "segment", "regions", "groups"]


@dataclass
class StageState:
    state: str = "idle"                        # idle | running | done | error | skipped
    progress: float = 0.0                      # 0..1
    message: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageInfo:
    width: int
    height: int
    work_width: int
    work_height: int
    preview_width: int
    preview_height: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ImageInfo":
        return _from_dict(cls, d)


@dataclass
class AnalysisOptions:
    detail: str = "balanced"                   # fast | balanced | max  (SAM point density / crops)
    intrinsic: str = "auto"                    # auto | careaga | heuristic
    max_groups: Optional[int] = None           # cap on color groups; None = automatic
    delta_e: float = 10.0                      # grouping threshold (CIEDE2000)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AnalysisOptions":
        return _from_dict(cls, d or {})
