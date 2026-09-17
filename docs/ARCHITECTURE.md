# Recolor — architecture and build contract

Photorealistic recoloring of product photographs: cars, motorcycles, sneakers, model
kits, furniture, anything with painted or dyed surfaces. The pipeline is modular, every
stage is inspectable and user-tunable, and the output keeps the exact lighting of the
original photograph because only the reflectance (albedo) layer is edited.

This document is the contract between the modules. Function names, signatures, file
layouts and JSON shapes below are binding; internals are free.

## 1. Pipeline

```
upload ─► ingest ─► intrinsic decomposition ─► segmentation ─► regions ─► color groups
                        (albedo / shading /      (SAM 2.1 + superpixels,   (cluster regions
                         residual, linear)        hierarchical merge)       by albedo)
prompt ─► palette (image search + k-means | parsed words | themes)
groups × palette ─► mapping (auto-suggest, user overrides)
albedo' = recolor(albedo, groups, mapping) ; out = sRGB(albedo' · shading + residual)
```

Resolutions (`recolor/config.py`):

| name     | long side | used for                                          |
|----------|-----------|---------------------------------------------------|
| original | as uploaded (≤ 6000) | export at full quality                  |
| work     | 1536      | intrinsic, SAM, regions, groups, all stored layers |
| preview  | 1024      | interactive renders                               |

Full-resolution export re-runs the intrinsic model on the original when it is at most
`FULLRES_INTRINSIC_MAX_PIXELS`; otherwise the working-res layers are guided-upsampled.
Label maps are always upsampled with `filters.upsample_labels` then snapped to edges with
`filters.refine_labels_with_guide`.

## 2. Shared foundation (already written — import, do not duplicate)

- `recolor/config.py` — paths, resolutions, device.
- `recolor/types.py` — `Region`, `ColorGroup`, `PaletteColor`, `PaletteSource`, `Palette`,
  `RenderOptions`, `AnalysisOptions`, `ImageInfo`, `StageState`, `Mapping` helpers, `STAGES`.
- `recolor/imageio.py` — `load_image`, `save_image`, `encode_png`, `encode_jpeg`,
  `save_f16`/`load_f16`, `fit_size`, `resize_to`, `resize_long_side`, `to_float`,
  `to_uint8`, `srgb_to_linear`, `linear_to_srgb` (gamma 2.2, matches Intrinsic),
  `rgb_to_lab`, `lab_to_rgb`, `linear_to_lab`, `lab_to_linear`, `delta_e` (CIEDE2000),
  `hex_to_rgb01`, `rgb01_to_hex`, `hex_to_lab`, `lab_to_hex`, `luminance`.
- `recolor/filters.py` — GPU `box_filter`, `guided_filter`, `gaussian_blur`,
  `upsample_labels`, `refine_labels_with_guide`, `soft_group_weights`.
- `recolor/colornames.py` — `nearest_name(lab)`, `hue_family(lab)`, `parse_color_words(prompt)`,
  `NAMED`, `WORD_COLORS`.

Conventions: RGB only, uint8 for display, float32 linear for intrinsic layers, int32 label
maps with no `-1` surviving a stage. Type hints and docstrings on public functions.

## 3. Module contracts

### 3.1 Intrinsic — `recolor/intrinsic/`  (owner: INT)

```python
# recolor/intrinsic/__init__.py
@dataclass
class IntrinsicResult:
    albedo: np.ndarray    # float32 linear HxWx3 in [0,1]
    shading: np.ndarray   # float32 linear HxWx3, >= 0 (may exceed 1)
    residual: np.ndarray  # float32 HxWx3, may be negative (saturated pixels) or positive (speculars)
    method: str           # 'careaga' | 'heuristic'

def decompose(image_rgb_u8: np.ndarray, method: str = "auto",
              progress: Callable[[float, str], None] | None = None) -> IntrinsicResult
def recompose(res: IntrinsicResult) -> np.ndarray   # uint8 sRGB
def warmup(method: str = "careaga") -> None          # load models onto the GPU (idempotent)
def is_loaded(method: str = "careaga") -> bool
def release(method: str = "careaga") -> None         # drop the model, free CUDA memory (idempotent)
```

Guarantees:
- `srgb_to_linear(image) == albedo * shading + residual` to within 1e-4 (both methods).
  The Careaga v2.1 pipeline already returns `lin_img = hr_alb * dif_shd + residual`;
  outputs are padded to a multiple of 32 by the model, so **crop back** to the input size.
- Output shape equals input shape (H, W).
- `method="auto"` uses Careaga when the model is available and the image is at most
  `config.FULLRES_INTRINSIC_MAX_PIXELS` pixels, else the heuristic.
- Heuristic (`recolor/intrinsic/heuristic.py`): edge-aware illumination estimate
  (guided-filtered luminance at a large radius, or multi-scale Retinex), shading
  grayscale replicated to 3 channels, `albedo = lin / shading` clipped to [0,1],
  `residual = lin - albedo*shading` so the identity holds exactly. Must run on a 1536
  image in under 0.5 s on the GPU.
- Careaga wrapper (`recolor/intrinsic/careaga.py`): lazy singleton via
  `intrinsic.pipeline.load_models('v2.1', device)`; `run_pipeline(models, img01, resize_conf=None)`
  keeps the input size. Measured: 0.7 s at 1536×1024 (4.8 GB), 3.5 s at 3072×2048 (13.8 GB).
  Use `torch.inference_mode()`. Catch CUDA OOM and fall back to the heuristic with a
  logged warning, never crash the job.
- Also provide `layers_for_display(res) -> dict[str, np.ndarray uint8]` with keys
  `albedo` (sRGB of albedo), `shading` (shading normalized by its 99.5th percentile,
  sRGB), `residual` (|residual| × 4, sRGB) for the UI.

Definition of done: `scripts/dev_intrinsic.py samples/motorcycle_1.jpg` writes the
three display layers plus a recomposition to `scratch/`, prints timings, VRAM and the
reconstruction error; `tests/test_intrinsic.py` covers the heuristic identity and the
crop-back logic (no model download in tests).

### 3.2 Segmentation — `recolor/segmentation/`  (owner: SEG)

The hard requirement: **very complex images** (a busy street, a sprue of 200 parts, a wall
of framed pictures) must come out fully labeled with sensible regions, and simple product
shots must not be over-fragmented.

```python
# recolor/segmentation/sam_masks.py
class SamMasker:            # lazy singleton, SAM 2.1 hiera-large from config.SAM2_CHECKPOINT
    def generate(self, image_rgb_u8: np.ndarray, detail: str = "balanced",
                 progress=None) -> list[dict]
    # each dict: {"segmentation": bool HxW, "area": int, "bbox": [x,y,w,h],
    #             "predicted_iou": float, "stability_score": float}
DETAIL_PRESETS = {"fast": {...}, "balanced": {...}, "max": {...}}
# points_per_side / crop_n_layers / pred_iou_thresh / stability_score_thresh /
# min_mask_region_area / use_m2m are the levers. "max" must still finish a 1536 image
# in under ~40 s on the 5090; report measured timings in the docstring.

# recolor/segmentation/superpixels.py
def slic_labels(image_rgb_u8: np.ndarray, n_segments: int, compactness: float = 10.0) -> np.ndarray  # int32

# recolor/segmentation/hierarchy.py
def build_regions(image_rgb_u8: np.ndarray, albedo_lin: np.ndarray, masks: list[dict],
                  detail: str = "balanced", progress=None) -> tuple[np.ndarray, list[dict]]
# -> (labels int32 HxW, every pixel in 0..N-1 ; per-region info dicts with
#     'source' ('sam'|'superpixel'|'split') and 'confidence')
```

`build_regions` algorithm (binding in spirit, tune freely):
1. Sort masks by area descending, paint largest first so smaller parts override wholes;
   drop masks below `min_area` (fraction of image, per preset) and masks whose mean
   albedo is within ΔE 4 of the mask they overlap by > 85 % (duplicate proposals).
2. Split any region whose albedo distribution is clearly bimodal (2-means in Lab on the
   region's albedo, keep the split if the two centroids differ by ΔE > 14 and both parts
   are above `min_area`) — SAM often returns one mask for a two-tone part.
3. Fill unlabeled pixels: SLIC superpixels on the image (n ∝ pixels / 400); each unlabeled
   superpixel becomes its own region, then merge each into the touching region with the
   nearest median albedo if ΔE < 8, else keep it.
4. Remove specks: regions below `min_area/4` merge into the touching region with the
   closest albedo. Relabel contiguous 0..N-1. Assert no `-1` remains.

```python
# recolor/segmentation/grouping.py
def group_regions(labels: np.ndarray, albedo_lin: np.ndarray, region_info: list[dict],
                  max_groups: int | None = None, delta_e: float = 10.0
                  ) -> tuple[list[Region], list[ColorGroup], np.ndarray]
# -> (regions, groups, group_map int32 HxW with values = group id in 0..G-1)
def regroup(regions: list[Region], labels, albedo_lin, max_groups, delta_e) -> same   # cheap re-cluster
def merge_groups(groups, regions, group_map, labels, ids: list[int]) -> same
def split_group(groups, regions, group_map, labels, albedo_lin, gid: int, k: int = 2) -> same
def move_regions(groups, regions, group_map, labels, region_ids: list[int], gid: int) -> same
```

Grouping: per-region median albedo in Lab (median, not mean — highlights and panel lines
skew means); area-weighted agglomerative clustering with CIEDE2000 linkage threshold
`delta_e`; if `max_groups` is set, keep merging the closest pair until the cap holds.
`ColorGroup.name = colornames.nearest_name(lab)`, `hue_family = colornames.hue_family(lab)`,
`is_background` = the group with the largest fraction of pixels on the image border, only
if that fraction is above 0.35 of the border. Groups sorted by area descending, ids 0..G-1.
Region ids stay stable across regroup/merge/split/move (only `group_id` changes) except
for split, which appends new region ids.

Definition of done: `scripts/dev_segment.py samples/street_complex_1.jpg --detail max`
writes `scratch/<name>_regions.png` (random colors), `_groups.png` (group albedo colors)
and `_edges.png` (boundaries over the image), prints region/group counts and timings for
`fast|balanced|max`; run it on at least `motorcycle_1`, `car_red_sports_1`,
`gundam_model_1`, `street_complex_1`, `interior_complex_1` and look at the outputs (Read
the PNGs). `tests/test_segmentation.py` covers `build_regions` on synthetic masks with
a hole and a duplicate, plus grouping/merge/split/move invariants — no SAM in tests.

### 3.3 Palette — `recolor/palette/`  (owner: PAL)

```python
# recolor/palette/__init__.py
def generate_palette(prompt: str, n_colors: int = 6, max_images: int = 6,
                     progress=None) -> Palette
# recolor/palette/sources.py
def search_images(prompt: str, limit: int) -> list[dict]   # {url, page_url, title, license, width, height}
def fetch_thumbs(items, cache_dir, width=640) -> list[tuple[dict, np.ndarray]]
# recolor/palette/extract.py
def extract_palette(images: list[np.ndarray], n_colors: int) -> list[PaletteColor]
# recolor/palette/themes.py
THEMES: dict[str, list[str]]         # ~40 curated themes, each 5–8 hexes
def match_theme(prompt: str) -> tuple[str, list[str]] | None
```

Sources: Wikimedia Commons search API (no key; use a descriptive User-Agent with the
project name), `filetype:bitmap`, prefer JPEG/PNG ≥ 800 px, thumbs at 640 px via
`iiurlwidth`. Optional second source via `ddgs` images if Commons returns < 3 results;
never fail the palette if the network fails — fall back to theme/parsed/default.
Cache per `sha1(prompt|n)` under `config.PALETTE_CACHE_DIR/<pid>/` with `palette.json`
and `sources/<i>.jpg`. Palette ids are that hash.

Extraction: pool ≤ 150k pixels per image, convert to Lab, weight pixels by chroma
(`1 + C/60`) so themes are not dominated by sky/gray, k-means with `k = n + 4`, merge
centroids within ΔE 9, drop clusters under 2 % weight, sort by weight, take `n`.
Ensure the result spans lightness: if all L are within 25, add the darkest and lightest
merged centroids back. Names via `colornames.nearest_name`.

Method rules: `parse_color_words` finds ≥ 2 explicit colors → `parsed` (image colors
appended after them up to `n`, method `mixed`); theme keyword match → theme colors first
then image colors; else `images`; nothing at all → `fallback` (a tasteful neutral+accent
set). Always return exactly `n_colors` colors (pad from theme/fallback) with weights
summing to 1.

Definition of done: `scripts/dev_palette.py "hawaii sunset"` prints the palette and
writes `scratch/palette_<pid>.png` swatch strip + source thumbs; `tests/test_palette.py`
covers extraction on synthetic images, theme matching, parsed prompts, and caching — no
network in tests (monkeypatch `search_images`).

### 3.4 Mapping — `recolor/mapping.py`  (owner: PAL)

```python
STRATEGIES = ["balanced", "area", "luminance", "hue", "contrast"]
def suggest_mapping(groups: list[ColorGroup], colors: list[PaletteColor] | list[str],
                    strategy: str = "balanced", keep_background: bool = True,
                    keep_locked: bool = True) -> Mapping
```

- `area`: rank groups by area, palette by weight, pair in order.
- `luminance`: pair by lightness rank (keeps the design's value structure).
- `hue`: Hungarian assignment on hue-angle distance; neutrals map to neutrals if any.
- `contrast`: Hungarian on |L_i − L_j| so light/dark relationships are preserved.
- `balanced` (default): Hungarian on `0.45·|area rank diff|/G + 0.45·|ΔL|/100 + 0.10·hue distance/180`.
When there are more groups than colors, assign the unique matches first, then each
remaining group takes the palette color nearest in Lab. Locked/background groups map
to `None` when the flags are set. Neutral groups (hue_family neutral, chroma < 9) prefer
the palette's neutral colors when the palette has any.

Tests in `tests/test_mapping.py`: uniqueness when G ≤ N, determinism, locked/background
handling, luminance ordering preserved by `contrast`.

### 3.5 Recoloring engine — `recolor/engine.py`  (owner: ENG)

```python
class Renderer:
    """Holds one image's layers on the GPU so successive renders are fast."""
    def __init__(self, albedo_lin: np.ndarray, shading_lin: np.ndarray, residual: np.ndarray,
                 group_map: np.ndarray, groups: list[ColorGroup]): ...
    def render(self, mapping: Mapping, options: RenderOptions) -> np.ndarray   # uint8 sRGB
    def render_at(self, long_side: int, mapping, options) -> np.ndarray         # resized layers, cached per size
def render_once(albedo_lin, shading_lin, residual, group_map, groups, mapping, options) -> np.ndarray
```

Algorithm (torch, GPU):
1. Coverage `m`: the fraction of each pixel belonging to a repainted group. A symmetric
   blur of the mapped indicator is wrong — it dips below 1 inside the part, leaving a rim
   of the old paint at every boundary with an unrepainted group. Snap the indicator to the
   photograph's edges with `filters.guided_filter_color`, max with the hard label, then
   ramp outward only. `feather_px` scales the outward ramp.
2. For each mapped group g with target hex T and source albedo `A_g` (group `albedo_lab`):
   - `shift` mode: chroma moves so A lands on T, `ab' = T_ab + s·R(θ)·(ab − A_ab)` with
     `s = min(1, C_T/C_A)` and `R(θ)` the rotation from the source's hue to the target's
     (faded out when either side is neutral). The deviation from the group colour is the
     paint's texture in the *source's* a/b frame; unrotated it lands beside the new colour
     instead of along it, which made the lit flank of a red tank painted navy come out
     mauve and its white decal cyan. Lightness uses a map anchored on the group's own lightness,
     `L' = T_L + slope·(L − A_L)`, slope 1 above the anchor and `T_L/A_L` below it. A plain
     additive shift pushes the darker half past zero, collapsing it to featureless black
     whenever the target is dark. `flat` mode: `alb_lab' = T_lab`; `texture` blends.
   - `saturation` scales the target's chroma before use.
   - Do the Lab math in torch; do not round-trip through skimage per render.
3. `albedo' = Σ_g W_g · f_g(albedo)` (unmapped groups: identity).
4. `shading' = shading ** shading_strength` (per-channel, keeps color of light).
5. `residual'`: when `keep_residual`, add it, but not untouched. The positive residual
   carries diffuse energy in the *original* paint's colour as well as specular light, so
   it must not survive a repaint unchanged — that is what made a red part painted black
   come out maroon. Split it at its achromatic floor: rebuild the coloured excess as a
   multiple of the repainted product (same energy, new colour), and keep the neutral floor
   where it is a sharp glint while attenuating it with the repaint where it is a faint
   veil. `residual_tint` lerps what survives toward the target colour.
6. The light on a repainted part changes colour with the paint. Diffuse shading is supposed
   to carry only the illuminant, but the decomposition leaks part of a saturated surface
   into it, most in shadows and concavities (37 % redder under a red shield, 5x redder than
   blue in the Ducati tank's shadows): that is bounce off the old paint. Per pixel, the
   light's tint beyond the scene's illuminant (estimated from well-lit low-chroma surfaces)
   is split along the old paint's chroma direction; the aligned part is rotated and scaled
   to the new paint like the albedo, the rest is kept. Removing only each group's median
   leak left the shadows of a navy repaint teal.
7. `out = linear_to_srgb(clip(albedo' · shading' + residual', 0, 1))` → uint8.

Performance target: preview (1024 long side) render < 60 ms after the first call;
full 4K render < 1 s. `tests/test_engine.py`: identity mapping reproduces the
recomposed original (max abs error ≤ 2/255), a flat repaint of one group has that
group's mean albedo at the target (ΔE < 3), feathering keeps outputs finite, all
options accepted.

`scripts/dev_render.py` takes an analyzed job directory (see §4) and a mapping JSON,
writes `scratch/render_*.png`. Until a job exists it must also work with
`--synthetic` (heuristic intrinsic + SLIC grouping on a sample image) so the engine
can be validated standalone.

### 3.6 Pipeline and jobs — `recolor/pipeline.py`, `recolor/jobs.py`  (owner: SRV)

```python
# recolor/jobs.py
class Job:                     # one uploaded image and everything derived from it
    id: str; dir: str; meta: dict     # meta is exactly the JSON returned by GET /api/jobs/{id}
    def save(self); def path(self, *parts) -> str
    def set_stage(self, stage, state, progress=None, message=None)   # publishes an event
    def subscribe(self) -> queue.Queue; def unsubscribe(self, q)
    def publish(self, event: dict)
class JobRegistry:              # loads data/jobs/*/job.json on start; thread-safe
    def create(self, image_rgb_u8, name, options: AnalysisOptions) -> Job
    def get(self, id) -> Job | None; def list(self) -> list[dict]; def delete(self, id)
registry = JobRegistry()

# recolor/pipeline.py
def analyze(job: Job) -> None            # runs in a worker thread; fills stages, writes artifacts
def load_layers(job: Job) -> dict        # cached: albedo/shading/residual/labels/group_map arrays
def get_renderer(job: Job) -> Renderer   # cached per job, invalidated on regroup/merge/split/move
def render_preview(job, mapping, options) -> bytes   # JPEG at preview res
def export(job, mapping, options, quality: str, fmt: str) -> dict  # {file, width, height, ms}
def apply_group_edit(job, kind: str, payload: dict) -> None   # merge|split|move|regroup|update
```

`analyze` order: ingest (save original, make work + preview images) → intrinsic →
segment (SAM) → regions → groups; each stage sets progress and a human message
("Finding parts with SAM 2 · 64 points/side · crop layer 1/2"), catches exceptions into
`status: "error"` with the message, and records seconds. A single worker thread
processes jobs FIFO (the GPU is shared). SAM and Intrinsic are lazy singletons: by
default they load on the first job that needs them and a background watchdog drops
them again after `config.IDLE_UNLOAD_S` (120 s) with none running, so the GPU only
holds their ~2-3 GB while the app is in active use; `serve.py --warmup` preloads both
at startup instead, but they are still unloaded on the same idle timer afterwards.

Job artifacts under `data/jobs/<id>/`:

```
job.json          # meta (see API)          original.jpg   # full res (png if upload was png)
work.png          # working res             preview.jpg    # preview res
albedo.npy shading.npy residual.npy         # float16, working res, linear
labels.npy group_map.npy                    # int32, working res
layers/albedo.jpg layers/shading.jpg layers/residual.jpg layers/regions.png
layers/groups.png layers/edges.png          # display layers, working res
ids/regions.png   # RGB-encoded region ids: id = R + 256·G + 65536·B
ids/groups.png    # R = group id (0..255), G = B = 0
exports/<name>.png|jpg
```

### 3.7 HTTP API — `recolor/server/app.py`, `serve.py`  (owner: SRV)

FastAPI. Static `web/` mounted at `/` (index.html at `/`, no caching in dev). JSON
errors as `{"error": "...", "detail": "..."}` with proper status codes. CORS open.

| method | path | body / query | returns |
|---|---|---|---|
| GET | `/api/health` | | `{ok, device, gpu, vram_total_mb, vram_used_mb, models:{sam2, intrinsic} ('cold'|'loading'|'ready'), jobs, version}` |
| GET | `/api/samples` | | `[{name, url:"/api/samples/<name>", thumb:"/api/samples/<name>?w=320", width, height, title, license}]` |
| GET | `/api/samples/{name}` | `?w=` optional | image |
| GET | `/api/jobs` | | `[JobSummary]` newest first: `{id, name, created, status, thumb, width, height, n_groups}` |
| POST | `/api/jobs` | multipart `file` **or** JSON `{sample: name}`; optional fields `detail`, `intrinsic`, `max_groups`, `delta_e` | `Job` (status queued) |
| GET | `/api/jobs/{id}` | | `Job` |
| DELETE | `/api/jobs/{id}` | | `{ok}` |
| GET | `/api/jobs/{id}/events` | SSE | events `{type:"stage", stage, state, progress, message}`, `{type:"status", status}`, `{type:"groups", groups}`, `{type:"done"}`, `{type:"error", message}`; replays the current state on connect; heartbeat comment every 15 s |
| GET | `/api/jobs/{id}/layers/{layer}` | layer ∈ `original, work, preview, albedo, shading, residual, regions, groups, edges` | image |
| GET | `/api/jobs/{id}/ids/{kind}` | kind ∈ `regions, groups` | PNG (see §3.6) |
| POST | `/api/jobs/{id}/groups/merge` | `{group_ids:[..]}` | `Job` |
| POST | `/api/jobs/{id}/groups/split` | `{group_id, k}` | `Job` |
| POST | `/api/jobs/{id}/groups/move` | `{region_ids:[..], group_id}` | `Job` |
| PATCH | `/api/jobs/{id}/groups/{gid}` | `{name?, locked?, is_background?}` | `Job` |
| POST | `/api/jobs/{id}/regroup` | `{max_groups?, delta_e?}` | `Job` |
| POST | `/api/palettes` | `{prompt, n_colors}` | `Palette` |
| GET | `/api/palettes/{pid}` | | `Palette` |
| GET | `/api/palettes/{pid}/sources/{i}.jpg` | | image |
| POST | `/api/jobs/{id}/mapping/suggest` | `{colors:[hex..], strategy}` | `{mapping:{gid:hex|null}}` |
| POST | `/api/jobs/{id}/render` | `{mapping, options}` | `image/jpeg` preview, header `X-Render-Ms` |
| POST | `/api/jobs/{id}/export` | `{mapping, options, quality:"work"|"full", format:"png"|"jpg"}` | `{url, width, height, ms}` |
| GET | `/api/jobs/{id}/exports/{file}` | | file, `Content-Disposition: attachment` |
| PUT | `/api/jobs/{id}/state` | `{mapping?, render_options?, palette_id?}` | `Job` (persists UI state) |

`Job` JSON:

```json
{"id":"a1b2c3d4e5f6","name":"motorcycle_1.jpg","created":1757700000.0,
 "status":"queued|analyzing|ready|error","error":null,
 "image":{"width":3067,"height":2045,"work_width":1536,"work_height":1024,"preview_width":1024,"preview_height":683},
 "options":{"detail":"balanced","intrinsic":"auto","max_groups":null,"delta_e":10.0},
 "stages":{"ingest":{"state":"done","progress":1,"message":"","seconds":0.4}, "intrinsic":{...},"segment":{...},"regions":{...},"groups":{...}},
 "timings":{"total_s":9.8},
 "intrinsic_method":"careaga",
 "groups":[ColorGroup...], "regions_count":212,
 "palette_id":null, "mapping":{}, "render_options":{}}
```

`serve.py`: `uvicorn` on `0.0.0.0:config.SERVER_PORT`, prints the LAN and Tailscale
URLs (detect like SaxScope's `serve.py`: UDP-connect trick for LAN, `tailscale ip -4`),
starts the idle-unload watchdog (models load lazily unless `--warmup` is passed), and
mentions `sudo ufw allow <port>/tcp` if the port is not reachable from the LAN.
`tests/test_server.py` uses FastAPI's TestClient
with the pipeline monkeypatched (no models): job create from a sample, events replay,
palette endpoint with a stubbed `search_images`, render with a stub renderer.

### 3.8 Frontend — `web/`  (owner: WEB)

No build step: `web/index.html`, `web/app.css`, `web/js/*.js` ES modules. Must look
and feel like a shipped product, not a prototype: a real brand, a coherent design
system, deliberate motion, empty states, loading states, keyboard shortcuts, and no
raw JSON anywhere in the UI.

Brand: **Chroma Studio** (wordmark in `index.html`, favicon as inline SVG data URI).
Typography from Google Fonts: `Sora` (600/700) for display, `Inter` (400/500/600) for
UI, `JetBrains Mono` for values. Dark theme by default with a light theme toggle; both
palettes as CSS custom properties on `:root` / `:root[data-theme="light"]`.

Views (single page, hash-routed `#/`, `#/studio/<jobId>`, `#/gallery`, `#/how`):

1. **Home** — hero with the wordmark, one-line promise, a large drop zone (drag/paste/
   click, shows a live preview + "Analyze" with a detail selector Fast/Balanced/Max),
   and a "Try a sample" strip from `/api/samples`. Recent jobs row from `/api/jobs`.
2. **Studio** — the workspace:
   - Left: canvas viewer. Layer tabs `Result · Original · Albedo · Shading · Regions ·
     Groups`. Before/after wipe slider (drag handle, spring easing) when on Result.
     Pan/zoom (wheel + drag, pinch), fit/1:1 buttons. **Hover highlights the region
     under the cursor** (decode `ids/regions.png` and `ids/groups.png` into typed arrays
     once; draw a glow outline with a second canvas); click selects the group; shift-
     click adds regions to a multi-selection for "move to group". Group selection shows
     a floating pill with the group's name, area %, and quick actions.
   - Right: inspector with collapsible panels and a progress **stepper** during
     analysis (five steps, animated progress bars, messages from SSE, elapsed time).
     - *Groups*: list of swatches (albedo color, name, area bar, region count), lock
       toggle, background badge, rename inline, drag one group onto another to merge,
       "Split" and "Auto-regroup" (slider for group count) actions.
     - *Palette*: prompt input with suggestions ("Hawaii sunset", "Stealth matte",
       "Sakura", "Racing livery", "Cyberpunk", "Desert camo"), count selector, Generate;
       shows source thumbnails (attribution on hover) and swatches that animate in;
       add/remove/edit colors (native color input), reorder by drag.
     - *Mapping*: strategy select (Balanced/Area/Luminance/Hue/Contrast) + "Suggest";
       rows `group swatch → target swatch`, drag any palette swatch onto a row, clear
       to keep original, per-row color picker. Any change re-renders the preview
       (debounced 80 ms, in-flight request cancelled with AbortController, show a
       thin progress bar at the top of the canvas, display `X-Render-Ms`).
     - *Finish*: realism controls (Texture, Feather, Shading strength, Highlight tint,
       Saturation) as sliders with live preview, then Export (Work/Full, PNG/JPG) with a
       download button and a "Copy share link".
3. **Gallery** — cards of all jobs with thumbnails, status chips, delete.
4. **How it works** — animated pipeline diagram (five stages light up in sequence),
   one paragraph each, no jargon soup.

Motion: enter transitions (fade + 8 px rise, 240 ms, `cubic-bezier(.2,.8,.2,1)`),
FLIP reorder for lists, skeleton shimmer while loading, count-up numbers, the wipe
handle with spring easing, hover glow on regions, toast notifications bottom-right,
`prefers-reduced-motion` respected everywhere. Keyboard: `1–6` layer tabs, `space`
hold to peek original, `⌘/Ctrl+Z` undo mapping change, `Esc` clear selection.

Implementation: `js/app.js` (router + boot), `js/api.js` (fetch wrappers + SSE +
abortable render), `js/state.js` (tiny observable store with undo stack for the
mapping), `js/viewer.js` (canvas, layers, hover/select, wipe, zoom), `js/panels/*.js`
(groups, palette, mapping, finish, stepper), `js/motion.js` (helpers), `js/toast.js`,
`js/util.js` (color math for swatches: hex↔rgb, contrast text color). No frameworks;
the DOM is built with small template helpers. Everything the API returns is rendered
through these panels — no `JSON.stringify` in the UI.

While the backend is being built in parallel, develop against `web/mock/` : a
`mock-server.py` (stdlib `http.server`) that serves the static files and fakes the
API with the sample images and made-up groups/palettes, including SSE progress.
Delete nothing when the real server arrives; the mock stays for UI work.

Definition of done: opens without console errors on Chrome; Home → drop or sample →
stepper animates → Studio shows layers, hover highlight, palette generation, mapping
drag, live render, export; responsive down to a 1024 px window; no layout shift on
load; Lighthouse-style basics (labels on inputs, focus rings, contrast).

## 4. Ownership map (parallel build)

| owner | writes only |
|---|---|
| INT | `recolor/intrinsic/*`, `scripts/dev_intrinsic.py`, `tests/test_intrinsic.py` |
| SEG | `recolor/segmentation/*`, `scripts/dev_segment.py`, `tests/test_segmentation.py` |
| PAL | `recolor/palette/*`, `recolor/mapping.py`, `scripts/dev_palette.py`, `tests/test_palette.py`, `tests/test_mapping.py` |
| ENG | `recolor/engine.py`, `scripts/dev_render.py`, `tests/test_engine.py` |
| SRV | `recolor/jobs.py`, `recolor/pipeline.py`, `recolor/server/*`, `serve.py`, `tests/test_server.py` |
| WEB | `web/**` |

Shared modules (§2) are frozen during the parallel build; if one needs a change,
add a helper in your own module and note it in your report. Cross-module imports
follow this contract; if a sibling module is not there yet, write against the
signature and stub it in your tests.

## 5. Verification

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/dev_intrinsic.py samples/motorcycle_1.jpg
.venv/bin/python scripts/dev_segment.py samples/street_complex_1.jpg --detail max
.venv/bin/python scripts/dev_palette.py "hawaii sunset"
.venv/bin/python scripts/dev_render.py --synthetic samples/car_red_sports_1.jpg
.venv/bin/python serve.py
```
