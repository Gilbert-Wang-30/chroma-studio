"""Curated, designed color themes and keyword matching against prompts.

Every theme is a hand-picked list of 5-8 sRGB hexes ordered from the most important
(dominant) color to the accent, so that a mapping strategy that ranks palette colors by
weight paints the biggest surfaces with the theme's signature color.  Themes are meant
to look designed: each one has a value structure (a dark, a mid, a light) and at most
two accents, never a random scatter of hues.
"""
from __future__ import annotations

import re

# name -> ordered hexes (dominant first).  Names are lowercase and can be typed by users.
THEMES: dict[str, list[str]] = {
    # ---- landscapes and moods
    "hawaii sunset":     ["#ff7b39", "#ffb347", "#ff4f6d", "#c43a9e", "#5b2a86", "#1c1b4a", "#ffe3b3"],
    "sunset":            ["#ff7a59", "#ffb35c", "#d4457a", "#7a2a7c", "#2d1b4e", "#ffe29a"],
    "tropical":          ["#00a896", "#ff7f11", "#ff206e", "#02c39a", "#028090", "#f0f3bd"],
    "ocean":             ["#0a4d68", "#088395", "#05bfdb", "#03253e", "#8fe3f0", "#f2f7f9"],
    "coral reef":        ["#ff7f50", "#2ec4b6", "#ff5c8a", "#ffd166", "#0a7c8c", "#fff1e6"],
    "forest":            ["#1f5f2a", "#3f8a3e", "#0b2e13", "#7fb069", "#5c4a32", "#b5cf8a", "#e9e4d1"],
    "autumn":            ["#9c2f1c", "#d3542b", "#e88a2a", "#5b1f0f", "#f2c14e", "#6b4a2b", "#f5ead6"],
    "arctic":            ["#dbe9f4", "#a9c6dd", "#f8fbff", "#6f9ac6", "#3a5f8a", "#1d2f4a"],
    "desert":            ["#e2c290", "#c19a6b", "#a67b5b", "#f1e7cf", "#7a6a53", "#3f7fa6"],
    "sand and sea":      ["#f3e9d2", "#e2c290", "#8fb3c9", "#3f7fa6", "#c9a86a", "#6b5a44"],
    "lava":              ["#e03e1f", "#8a1e0f", "#ff8c1a", "#0e0e0e", "#3a1a12", "#ffd166"],
    "galaxy":            ["#141a4d", "#3b2a78", "#6d3fa0", "#050716", "#c0399d", "#f0c4ff", "#ffffff"],
    "midnight":          ["#1b2140", "#0b1026", "#2e3a63", "#4d5e8f", "#8fa2cf", "#e6ebf7"],
    "sakura":            ["#f9c8d6", "#f4a6c1", "#fde2e8", "#e07a9a", "#7b4b63", "#4a3b44", "#f7f2ef"],
    "lavender":          ["#c9b6e4", "#a98bd3", "#e8dff5", "#7f5fb3", "#4d3a7a", "#f8f6fc"],
    "mint":              ["#b2f2d9", "#6fd8b0", "#2ea882", "#e6fff5", "#17624b", "#f8fff9"],
    "sunflower":         ["#f7d51d", "#ffb000", "#e07b00", "#2f5d2a", "#4b3b1a", "#fffbe6"],
    # ---- automotive and motorsport
    "racing livery":     ["#d62828", "#f7f4ef", "#111111", "#ffcc00", "#1f4fd1", "#8a8f96"],
    "gulf racing":       ["#7fc0e6", "#f58a07", "#f5f2ea", "#1e2a44", "#c9cdd2", "#101010"],
    "martini racing":    ["#f4f1ea", "#1c3f94", "#79b4e5", "#c8102e", "#0d1b3d", "#c0c0c0"],
    "jdm":               ["#1f5fb6", "#f5f4ee", "#c8102e", "#2b1b4e", "#ff8c1a", "#101010"],
    "stealth matte":     ["#141416", "#24272b", "#363b41", "#4d545c", "#6f767f", "#8f1d1d"],
    "carbon":            ["#1c1f23", "#2f343a", "#4a5058", "#0b0c0e", "#7a828c", "#c3c8cf"],
    "chrome":            ["#dbe4eb", "#b9c3cc", "#8f9aa5", "#5f6a75", "#2e3540", "#f7fafc"],
    "rosso corsa":       ["#d40000", "#ff2800", "#7a0000", "#1a1a1a", "#ffd700", "#f4f4f4"],
    "british racing green": ["#004225", "#1e5c3c", "#3d7a5a", "#c9a45c", "#f3ede0", "#1a1a1a"],
    "supercar":          ["#ffd400", "#ff5f00", "#39ff14", "#1a1a1a", "#c8c8c8", "#f4f4f4"],
    "bumblebee":         ["#ffd400", "#1a1a1a", "#3a3a3a", "#f4f4f4", "#8a7a2a", "#b0b0b0"],
    "denim":             ["#2f5187", "#1c2f4f", "#5a7db8", "#9bb3d1", "#e8ecf2", "#c2a26b"],
    # ---- military and utility
    "military olive":    ["#4b5320", "#3b3f24", "#6b7042", "#8a8f5a", "#b3ad8c", "#2b2b25", "#c8b27a"],
    "desert camo":       ["#c19a6b", "#e2c290", "#a67b5b", "#7a6a53", "#5a5040", "#ede2c4"],
    "woodland camo":     ["#5a6b3f", "#3b4a2a", "#6b5433", "#8a7f5c", "#2a2b23", "#b0a884"],
    "urban camo":        ["#5a5a5a", "#8f8f8f", "#2b2b2b", "#c4c4c4", "#eeeeee", "#3d4750"],
    "safety orange":     ["#ff6a00", "#1a1a1a", "#f4f4f4", "#ffcc00", "#3d4750", "#8a8f96"],
    # ---- neon, retro, pop
    "cyberpunk":         ["#ff2a6d", "#05d9e8", "#0d0221", "#261447", "#fcee0a", "#7a04eb", "#d1f7ff"],
    "neon tokyo":        ["#ff3caf", "#00e5ff", "#0b0b1f", "#1b1b3a", "#ffe600", "#7a5cff", "#ff6a00"],
    "vaporwave":         ["#ff71ce", "#01cdfe", "#b967ff", "#05ffa1", "#fffb96", "#2a1b3d", "#f0e6ff"],
    "miami":             ["#ff6ec7", "#00e0ff", "#ffcf5c", "#ffffff", "#2b1b4e", "#ff8c69"],
    "candy":             ["#ff4d8d", "#ff9ecb", "#ffd166", "#7cf0ba", "#4ecdc4", "#a78bfa", "#fff7fb"],
    "pastel":            ["#ffd1dc", "#aec6cf", "#c1e1c1", "#fff5ba", "#e3c9f7", "#ffdab9", "#5f6672"],
    "retro 70s":         ["#e0a526", "#c8631f", "#8a3d1c", "#5b6e3a", "#d9c8a0", "#3a2a1e"],
    "punk":              ["#0f0f0f", "#ff1dce", "#2b2b2b", "#ff69b4", "#ffd1dc", "#f4f4f4"],
    "rainbow":           ["#e40303", "#ff8c00", "#ffed00", "#008026", "#004dff", "#750787"],
    "teal and orange":   ["#0d6e7c", "#ff8c42", "#0a3d4a", "#3fb8c9", "#c95a2a", "#f5e6d3"],
    # ---- luxury and interior
    "royal":             ["#2b1a5c", "#4169e1", "#d4af37", "#7a0f2d", "#f6f0e8", "#1a1a2e"],
    "navy and gold":     ["#1a2a5a", "#0f1e3d", "#d4af37", "#2c4a8a", "#f0d27a", "#f5f1e6"],
    "black and gold":    ["#0c0c0c", "#d4af37", "#1e1a14", "#7a5c1e", "#f2d98d", "#f6f1e5"],
    "rose gold":         ["#b76e79", "#d69aa5", "#e8c4c4", "#f3e5e1", "#7d4a52", "#3b2a2e"],
    "emerald":           ["#146c4c", "#1f9a6e", "#0b3d2e", "#0f5e8a", "#7b1e4e", "#d4af37"],
    "burgundy":          ["#6b1020", "#8f1d30", "#3d0c11", "#b03a48", "#d98b93", "#f0e2df"],
    "tiffany":           ["#81d8d0", "#0abab5", "#c8f0ee", "#f4f4f4", "#2b2b2b", "#d4af37"],
    "monochrome":        ["#2b2b2b", "#0f0f0f", "#555555", "#8a8a8a", "#bfbfbf", "#f2f2f2"],
    "nordic":            ["#f7f5f0", "#dcd6cc", "#a8a196", "#6b6a66", "#2f3437", "#c1a27a"],
    "mocha":             ["#6f4a35", "#4a2f24", "#a67b5b", "#2b1d16", "#d9b99b", "#f2e8dc"],
    "terracotta":        ["#c2703d", "#8c4a2f", "#e0a874", "#a48b6b", "#5e5245", "#f1e6d6"],
    "vintage":           ["#d9c9a5", "#f5efe0", "#b28a5c", "#8a5a3b", "#3d3128", "#7a8a7f"],
    "steampunk":         ["#6b4b2c", "#b87333", "#b5a642", "#3a2a1e", "#8b8c89", "#d9c7a5"],
}

# Extra words and phrases that select a theme.  A phrase matches only as whole words,
# and the longest matching phrase wins so "hawaii sunset" beats "sunset".
THEME_KEYWORDS: dict[str, list[str]] = {
    "hawaii sunset": ["hawaii", "hawaiian", "maui", "aloha", "waikiki", "hawaiian sunset"],
    "sunset": ["dusk", "golden hour", "sundown", "evening sky"],
    "tropical": ["tropics", "jungle", "caribbean", "bahamas", "tiki", "palm"],
    "ocean": ["sea", "deep sea", "marine", "underwater", "aquatic", "nautical", "seaside"],
    "coral reef": ["reef", "coral"],
    "forest": ["woods", "woodland", "pine", "moss", "rainforest", "evergreen"],
    "autumn": ["fall", "fall leaves", "harvest", "maple", "pumpkin spice"],
    "arctic": ["polar", "ice", "frost", "glacier", "snow", "winter", "icy", "frozen"],
    "desert": ["dune", "dunes", "sahara", "canyon", "mesa", "southwest"],
    "sand and sea": ["beach", "coastal", "seaside beach", "shore"],
    "lava": ["volcano", "volcanic", "magma", "ember", "embers", "inferno", "fire"],
    "galaxy": ["space", "nebula", "cosmic", "cosmos", "outer space", "milky way", "stars", "aurora"],
    "midnight": ["night", "moonlight", "nocturne", "deep night", "midnight blue"],
    "sakura": ["cherry blossom", "cherry blossoms", "blossom", "hanami", "japanese spring"],
    "lavender": ["lilac", "wisteria", "provence"],
    "mint": ["fresh mint", "spearmint", "seafoam"],
    "sunflower": ["sunny", "sunshine", "honey bee", "lemon yellow"],
    "racing livery": ["racing", "race car", "livery", "motorsport", "rally", "le mans", "touring car", "gt3"],
    "gulf racing": ["gulf", "gulf livery", "gulf oil", "gulf blue"],
    "martini racing": ["martini", "martini livery", "martini stripes"],
    "jdm": ["japanese domestic", "tuner", "drift", "initial d", "nismo", "bayside blue"],
    "stealth matte": ["stealth", "murdered out", "blackout", "all black", "satin black", "black ops"],
    "carbon": ["carbon fiber", "carbon fibre", "graphite", "titanium"],
    "chrome": ["silver", "metallic", "brushed metal", "aluminium", "aluminum", "steel"],
    "rosso corsa": ["ferrari", "ferrari red", "italian red", "rosso", "scuderia", "corsa"],
    "british racing green": ["racing green", "brg", "heritage", "bentley", "jaguar green", "classic british"],
    "supercar": ["lamborghini", "exotic", "hypercar", "lambo", "mclaren"],
    "bumblebee": ["bee", "wasp", "hornet", "yellow and black", "black and yellow", "taxi"],
    "denim": ["jeans", "indigo denim", "workwear"],
    "military olive": ["military", "army", "olive drab", "tank", "soldier", "field gear", "armed forces"],
    "desert camo": ["desert camouflage", "tan camo", "sand camo", "desert storm", "khaki camo"],
    "woodland camo": ["camo", "camouflage", "forest camo", "green camo", "hunting"],
    "urban camo": ["urban camouflage", "gray camo", "grey camo", "snow camo", "arctic camo", "digital camo"],
    "safety orange": ["hi vis", "hi-vis", "high visibility", "construction", "hazard", "blaze orange"],
    "cyberpunk": ["cyber", "blade runner", "night city", "hacker", "dystopia", "dystopian", "futuristic"],
    "neon tokyo": ["tokyo", "neon", "shibuya", "shinjuku", "akihabara", "arcade", "neon lights"],
    "vaporwave": ["vapor wave", "synthwave", "outrun", "retrowave", "aesthetic", "80s", "eighties"],
    "miami": ["miami vice", "south beach", "flamingo", "art deco"],
    "candy": ["sweets", "bubblegum", "lollipop", "gummy", "candy shop", "sugar"],
    "pastel": ["pastels", "soft colors", "soft colours", "baby colors", "easter", "kawaii"],
    "retro 70s": ["70s", "seventies", "retro", "groovy", "disco", "vintage 70s"],
    "punk": ["punk rock", "goth", "gothic", "hot pink and black", "emo"],
    "rainbow": ["pride", "spectrum", "multicolor", "multicolour", "colorful", "colourful"],
    "teal and orange": ["cinematic", "movie look", "blockbuster", "film look", "orange and teal"],
    "royal": ["regal", "king", "queen", "crown", "majestic", "imperial", "royalty", "purple and gold"],
    "navy and gold": ["gold and navy", "navy gold", "naval", "admiral"],
    "black and gold": ["gold and black", "luxe", "luxury", "opulent", "gold trim", "noir gold"],
    "rose gold": ["blush gold", "copper rose", "pink gold", "rose and gold"],
    "emerald": ["jewel", "jewel tones", "gemstone", "gem", "sapphire", "ruby"],
    "burgundy": ["wine", "merlot", "bordeaux", "oxblood", "maroon", "cherry"],
    "tiffany": ["tiffany blue", "robin egg", "robin's egg", "turquoise and gold"],
    "monochrome": ["black and white", "grayscale", "greyscale", "mono", "achromatic", "noir"],
    "nordic": ["scandinavian", "scandi", "minimal", "minimalist", "hygge", "ikea"],
    "mocha": ["coffee", "espresso", "latte", "cappuccino", "chocolate", "cocoa", "brown tones"],
    "terracotta": ["clay", "adobe", "tuscan", "earth tones", "earthy", "pottery"],
    "vintage": ["antique", "old school", "classic", "sepia", "aged", "heritage cream"],
    "steampunk": ["brass", "victorian", "clockwork", "copper and brass", "industrial"],
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> str:
    """Lowercase, fold punctuation to spaces, collapse whitespace; pad with spaces so a
    whole-word check is a plain substring test."""
    words = _TOKEN_RE.findall(text.lower().replace("-", " ").replace("&", " and "))
    return " " + " ".join(words) + " "


def theme_phrases() -> dict[str, list[str]]:
    """Every phrase (theme name plus keywords) that selects each theme, normalized."""
    out: dict[str, list[str]] = {}
    for name in THEMES:
        phrases = [name] + THEME_KEYWORDS.get(name, [])
        out[name] = [_normalize(p).strip() for p in phrases]
    return out


_PHRASES = theme_phrases()


def match_theme(prompt: str) -> tuple[str, list[str]] | None:
    """Find the curated theme a prompt refers to.

    Guarantees: returns ``(theme_name, hexes)`` with a *copy* of the theme's colors when
    any theme name or keyword appears in the prompt as whole words, or ``None``.  The
    longest matching phrase wins (more words, then more characters), so "hawaii sunset"
    selects the Hawaii theme rather than the generic sunset one; ties are broken by the
    order of ``THEMES``.  Matching is case- and punctuation-insensitive; ``None`` or an
    empty/blank prompt gives ``None`` rather than raising.
    """
    if not prompt or not isinstance(prompt, str):
        return None
    text = _normalize(prompt)
    if not text.strip():
        return None
    best: tuple[int, int, int, str] | None = None
    for order, (name, phrases) in enumerate(_PHRASES.items()):
        for phrase in phrases:
            if f" {phrase} " in text:
                key = (phrase.count(" ") + 1, len(phrase), -order, name)
                if best is None or key > best:
                    best = key
    if best is None:
        return None
    name = best[3]
    return name, list(THEMES[name])


def theme_names() -> list[str]:
    """All theme names in curated display order."""
    return list(THEMES.keys())
