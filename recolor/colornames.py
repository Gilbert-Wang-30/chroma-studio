"""Human names for colors, hue families, and parsing of color words in prompts."""
from __future__ import annotations

import math
import re

import numpy as np

from .imageio import delta_e, hex_to_lab, rgb_to_lab

# A curated, readable subset of CSS/X11 names plus a few paint-like names. Kept short on
# purpose: the nearest name should read naturally in a UI ("Crimson", not "LightGoldenrod3").
NAMED: dict[str, str] = {
    "Black": "#101010", "Charcoal": "#36454f", "Graphite": "#4a4a4a", "Slate": "#708090",
    "Gray": "#808080", "Silver": "#c0c0c0", "Light gray": "#d3d3d3", "Off-white": "#f2f0eb",
    "White": "#fafafa", "Ivory": "#fffff0", "Beige": "#f5f5dc", "Cream": "#fffdd0",
    "Sand": "#c2b280", "Tan": "#d2b48c", "Khaki": "#c3b091", "Olive": "#808000",
    "Olive drab": "#6b8e23", "Army green": "#4b5320", "Forest green": "#228b22",
    "Green": "#2e8b3d", "Lime": "#7fdd2a", "Mint": "#98ff98", "Sea green": "#2e8b57",
    "Teal": "#008080", "Turquoise": "#40e0d0", "Aqua": "#00e5e5", "Cyan": "#00bcd4",
    "Sky blue": "#87ceeb", "Steel blue": "#4682b4", "Royal blue": "#4169e1", "Blue": "#1f5fd6",
    "Cobalt": "#0047ab", "Navy": "#1a2a5a", "Midnight": "#191970", "Indigo": "#4b0082",
    "Violet": "#8f00ff", "Purple": "#7b2cbf", "Lavender": "#b57edc", "Plum": "#8e4585",
    "Magenta": "#ff00ff", "Fuchsia": "#e0218a", "Hot pink": "#ff69b4", "Pink": "#f4a6c1",
    "Rose": "#e8637a", "Salmon": "#fa8072", "Coral": "#ff7f50", "Crimson": "#dc143c",
    "Red": "#d62828", "Scarlet": "#ff2400", "Cherry": "#b0171f", "Maroon": "#7a1f2b",
    "Burgundy": "#800020", "Brick": "#9c3b2e", "Rust": "#b7410e", "Brown": "#7b4a2d",
    "Chocolate": "#5c3a21", "Copper": "#b87333", "Bronze": "#cd7f32", "Orange": "#ff8c1a",
    "Tangerine": "#f28500", "Amber": "#ffbf00", "Gold": "#d4af37", "Mustard": "#e1ad01",
    "Yellow": "#f7d51d", "Lemon": "#fff44f", "Peach": "#ffcba4", "Champagne": "#f7e7ce",
}

_NAMED_LAB = {n: np.array(hex_to_lab(h), np.float32) for n, h in NAMED.items()}
_NAMED_ARR = np.stack(list(_NAMED_LAB.values()))
_NAMED_KEYS = list(_NAMED_LAB.keys())


def nearest_name(lab) -> str:
    """Closest curated name to a Lab color (CIEDE2000)."""
    d = delta_e(np.repeat(np.asarray(lab, np.float32)[None], len(_NAMED_KEYS), 0), _NAMED_ARR)
    return _NAMED_KEYS[int(np.argmin(d))]


def hue_family(lab) -> str:
    """Coarse hue bucket used for display and for hue-aware mapping."""
    L, a, b = (float(v) for v in lab)
    chroma = math.hypot(a, b)
    if chroma < 9.0 or L < 8:
        return "neutral"
    h = (math.degrees(math.atan2(b, a)) + 360.0) % 360.0
    # Boundaries are on the CIE Lab hue circle (red ~35, yellow ~93, green ~143,
    # cyan ~217, blue ~291, purple ~314, magenta ~328), not the HSV one.
    if 5 <= h < 45:
        return "red"
    if 45 <= h < 75:
        return "orange"
    if 75 <= h < 115:
        return "yellow"
    if 115 <= h < 175:
        return "green"
    if 175 <= h < 250:
        return "cyan"
    if 250 <= h < 305:
        return "blue"
    if 305 <= h < 340:
        return "purple"
    return "magenta"


# Words a user might type; mapped to hexes so "navy and gold" yields a palette directly.
WORD_COLORS: dict[str, str] = {k.lower(): v for k, v in NAMED.items()}
WORD_COLORS.update({
    "grey": "#808080", "light grey": "#d3d3d3", "dark grey": "#4a4a4a", "dark gray": "#4a4a4a",
    "matte black": "#141414", "jet black": "#0a0a0a", "pearl white": "#f4f1ea", "gunmetal": "#2c3539",
    "racing green": "#004225", "british racing green": "#004225", "neon green": "#39ff14",
    "electric blue": "#0892d0", "baby blue": "#89cff0", "deep blue": "#0f2c7c", "ocean blue": "#1d6fa5",
    "sunset orange": "#fd5e53", "blood red": "#8a0303", "ferrari red": "#ff2800", "wine": "#722f37",
    "rose gold": "#b76e79", "brass": "#b5a642", "chrome": "#dbe4eb", "cyberpunk yellow": "#fcee0a",
    "hot magenta": "#ff1dce", "pastel pink": "#ffd1dc", "pastel blue": "#aec6cf", "pastel green": "#c1e1c1",
    "sakura": "#f9c8d6", "cherry blossom": "#ffb7c5", "military green": "#4b5320", "desert tan": "#c19a6b",
    "sand yellow": "#e2c290", "camo green": "#5a6b3f", "camo brown": "#6b5433", "arctic white": "#f8fbff",
})

_WORD_RE = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")


def parse_color_words(prompt: str) -> list[tuple[str, str]]:
    """Explicit colors mentioned in a prompt, in order: [(word, hex)]. Hex codes and
    multi-word names ("racing green") are matched before single words."""
    found: list[tuple[int, str, str]] = []
    for m in _WORD_RE.finditer(prompt):
        found.append((m.start(), m.group(0), m.group(0).lower()))
    low = prompt.lower()
    taken: list[tuple[int, int]] = []
    for word in sorted(WORD_COLORS, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(word) + r"\b", low):
            if any(s <= m.start() < e for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), word, WORD_COLORS[word]))
    found.sort()
    return [(w, h) for _, w, h in found]
