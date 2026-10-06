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
                        (albedo / shading /      (SAM 2.1 + superpixels,   (detected parts one group per
                         residual, linear)        hierarchical merge,       kind; backdrop and object clustered
                                                  recursive re-split,       apart, one paint under different
                                                  Florence-2 lettering      light merged, then refine: absorb,
                                                  and named parts, SAM      highlights, decals, ViTMatte edge
                                                  re-prompted on small      snap, locks, protect mask, finish
                                                  parts, BiRefNet matte,    badges; junk slivers folded in)
                                                  OWLv2 detected parts,
                                                  one subject kept)
prompt ─► palette (image search + k-means | parsed words | themes)
groups × palette ─► mapping (auto-suggest, user overrides; background ignored by default)
albedo' = recolor(albedo, groups, mapping) ; out = sRGB(albedo' · shading + residual)
```

Resolutions (`recolor/config.py`):

| name     | long side | used for                                          |
|----------|-----------|---------------------------------------------------|
| original | as uploaded (≤ 6000) | export at full quality                  |
| work     | 1536      | intrinsic, SAM, regions, groups, all stored layers |
| preview  | 1024      | interactive renders                               |

Full-resolution export re-runs the intrinsic model on the original when it is at most
`FULLRES_INTRINSIC_MAX_PIXELS` and the result agrees with the working-res albedo the preview
was tuned on (at most 10 % of the repainted pixels more than ΔE 10 apart; the yellow BMW's
full-res split drifted 19 % and its tail read flat); otherwise the working-res layers are
guided-upsampled. The pass's need is fitted to its measured peak allocations (1.75 GB + 1.9 GB
per megapixel: 20 GB for a 10 MP original); it may take at most 80 % of the free VRAM
straight away (read device-wide right before the pass, so other processes on the card count;
the rest stays free for other users' previews), otherwise the segmentation models (SAM 2,
ViTMatte, Florence-2, BiRefNet, OWLv2: about 6 GB, none of them needed by an export, all
reloaded lazily by the next analysis) are released first and the plain fit decides
(`pipeline._fullres_room`); only when it still does not fit are the working-res layers
upsampled. The caching allocator *reserves* more than the pass allocates (29 GB at 10 MP,
26 GB with expandable segments, which `serve.py` enables): where the free VRAM is below
that peak it frees its cached blocks and retries, which PyTorch logs as a
`CUDACachingAllocator` warning, and the pass completes. On a shared card (another server or
the owner's app holding a few GB) that warning is expected for a >= 9 MP export, not a
failure; gating the pass on the reserved peak instead would upsample nearly every 10 MP
export. The first
full-res pass stores its albedo, area-downsampled to the working
resolution (`fullres_albedo_<method>.npy`), so a later export whose mapping is known to
drift skips the 4-11 s, 20 GB pass instead of running it and throwing it away.
The group map is upsampled with `filters.upsample_labels` then snapped to edges with
`filters.refine_labels_with_guide`, together with the engine masks (one combined label:
group, island, protect), and the engine's pixel distances are scaled from the working
resolution (`reference_long_side`) so the export matches the preview. The white paint's glints
(engine rule 7e) and every group's neutral-source weights are taken from the working-resolution
layers and handed to the export's renderer (`pipeline._working_reference`, `Renderer(glints=...,
neutral=...)`; the preview's cached renderer is used when it holds the export's own snapshot, its
edit generation, else one is built and freed before the full-resolution pass, and the glints only
when the mapping repaints a neutral source): measured again on the full-resolution layers, which the
decomposition splits differently, they were not the preview's (the Alpine's shoulder glint went,
twenty sill reflections came; a group's glint share, which decides whether a glossy grey is white
paint, moves with the residual). A card too full for that renderer leaves the export to measure its
own.

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
                  detail: str = "balanced", progress=None, extra=None) -> tuple[np.ndarray, list[dict]]
# -> (labels int32 HxW, every pixel in 0..N-1 ; per-region info dicts with
#     'source' ('sam'|'superpixel'|'split'|'small'|'text'|'wheel'|'named'|'prompt'|'part'|'kind'),
#     'confidence', 'exempt' (small / text), for a detected part (source 'kind', see below)
#     'part_kind', 'part_label', 'part_plural', 'part_instance', 'part_score' and, after the
#     matte, 'bg' (0 object, 1 border set, 2 matte; a detected part is always 0))
def cut_on_matte(labels, info, fg, albedo_lin=None) -> (labels, info, n_cut)   # the foreground matte's cut
# recolor/segmentation/florence.py     Florence-2-large, lazy singleton (local cache only)
def analyse(image_rgb_u8) -> {"ocr": [{quad, text}], "grounding": [{box, label}]} | None
def caption(image_rgb_u8) -> str | None                                         # picks the part vocabulary
# recolor/segmentation/smallparts.py   lettering and named parts as extra masks for build_regions
def find_extras(image_rgb_u8, albedo_lin, analysis, prompter) -> list[Extra]   # Extra: mask, source, labels, parts
#                                      detected parts (after the matte cut)
def find_kind_parts(image_rgb_u8, albedo_lin, fg, caption, detector, prompter, budget_s=None,
                    gates=DEFAULT_PART_GATES) -> (list[PartMask], report)
def stamp_parts(labels, info, parts, albedo_lab) -> (labels, info, report)
def attach_parts(parts) -> parts                                  # a mirror's stalk folded into the mirror
# recolor/segmentation/partdetect.py   OWLv2 (large, ensemble), lazy singleton (local cache only)
def detect(image_rgb_u8, phrases) -> [{box, phrase, score, det}] | None
# recolor/segmentation/wheels.py
def split_wheel(wheel_mask, image_rgb_u8, albedo_lab) -> (tyre, rim) | None
# recolor/segmentation/subject.py     the matte keeps one subject (a car, a motorcycle, a bicycle)
def other_objects(image_rgb_u8, fg, prompter, keep=()) -> bool HxW   # another object the matte took in
def whole_regions(labels, mask, keep=None) -> bool HxW              # regions half inside go whole
def cut_off(labels, info, mask, albedo_lin=None) -> (labels, info, n_cut)
# recolor/segmentation/foreground.py   BiRefNet_dynamic, lazy singleton (local cache only)
def fg_prob(image_rgb_u8) -> float32 HxW in [0, 1] | None
# recolor/segmentation/sam_masks.py
SamMasker.prompt_boxes(image_rgb_u8, jobs) -> list[list[dict]]   # box / point prompts on crops, batched per crop
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
5. Re-split: step 2 runs again (up to `RESPLIT_ROUNDS` = 3 passes) on the regions it
   produced. The smaller half of a split is otherwise never examined again, and a mixed
   region survives (the yellow BMW's nose shared one khaki region with every blown
   specular of the bike, so no colour rule could group it with the paint).

Small, thin and named parts (measured in round 3 on the six reference photos: achievable
part recall at IoU 0.5 0.753 -> 0.826, thin-part recall 0.211 -> 0.474, decals 0.571 ->
0.857, rubber 0.714 -> 1.0, about +1 s per image at Balanced, +28 % regions):
- **Distinct small proposals** (step 1): a proposal below the preset's `min_area` is still
  painted when it has at least `SMALL_MIN_PX` (60) pixels and its median albedo is more than
  `SMALL_RING_DE` (15) from its 3 px outer ring (an indicator lens, a decal on a sprue),
  source `'small'`, no duplicate test. A two-tone proposal one of whose tones is its
  surroundings (thin lettering with the paint showing between the strokes: the BMW's
  "S1000", which otherwise clustered with the gold parts and stayed yellow under a navy
  repaint) keeps only the other tone; the 2-means centroids and the ring decide, with no
  valley test, because lettering a few px tall is anti-aliased into a continuum. Such a
  region is *exempt*: never split (step 2), never merged away as a speck (step 4) and never
  eroded by the guided-filter snap. A remnant below the floor (the slivers of a word mask
  between its letters) loses the exemption and merges into the paint it is. Applies at
  every preset.
- **Lettering** (`smallparts.text_masks`, Balanced and Max): Florence-2's OCR quads (the
  image plus four 0.6 tiles, one batched call; strings with fewer than two ASCII letters or
  digits are texture) are merged, grown by 4 px and prompted as SAM boxes; SAM's mask is
  kept when it lies inside the quad, covers at most 90 % of it and differs from the rest by
  more than dE 15, else the minority 2-means albedo cluster of the quad when it is bimodal.
  Stamped after the fill over everything but other text, source `'text'`, exempt. Bounded
  work: the quads are merged on their bounding boxes (union-find: 300 quads in 10 ms where
  the pixel-pair loop took minutes), at most 48 lettering domains and 48 phrase boxes (the
  largest) are prompted, in chunks of 16 against a 6 s wall-clock budget
  (`pipeline.EXTRAS_BUDGET_S`) after which the remaining prompts are skipped and logged: a
  text-dense sheet (416 OCR quads) costs 1.6 s of prompts instead of holding the worker.
- **Named parts** (`smallparts.part_masks`, Balanced and Max): the phrase boxes of one
  grounding caption (tire, wheel rim, brake disc, fork, exhaust pipe, muffler, headlight,
  turn signal, mirror, logo, emblem, spoke) prompted as SAM boxes. A wheel box is tyre +
  rim to SAM: `wheels.split_wheel` fits the outer ellipse (RANSAC on the filled contour),
  casts 144 rays and fits the rim's lip ellipse to the outermost strong photo / albedo edge
  on each; tyre and rim are stamped only from regions that reach outside the wheel in the
  wheel's own colour or hold both parts (the disc, the sprocket and a gold caliper are
  kept); a wheel that does not split adds nothing. The outer ellipse passes on the share of
  the contour on it (35 %) or, with a quarter of the contour on it, on its angular support (an
  inlier in 60 % of its 5-degree sectors: the Ducati's rear wheel, notched deep by the
  swingarm, the chain guard and the fender, had 31 % of its contour on an ellipse supported all
  the way round but for the swingarm, and stayed in the swingarm's colour group); when the
  outermost strong edge per ray finds no lip, the lip fallback lets every edge peak of the
  outer band (0.7-0.95 of the radius, a quarter of the ray's strongest edge) compete and keeps
  the ellipse the most rays support (60 % of the rays with a peak and a third of all rays: the
  BMW's black front rim in its black tyre, whose lip is a weaker edge than the silver disc
  inside it on most rays). Any other phrase mask is stamped only
  where it leaves at least 10 % (and 64 px) of a region on each side and the sides differ
  by more than dE 8 in albedo or 12 in photo lightness. Sources `'wheel'` / `'named'`, not
  exempt; a stamped part under 150 px is dropped.
- Without Florence-2 (package or weights missing, a CUDA OOM, any model error) the stage
  runs without lettering and named parts, logged; `fast` never asks for them.

Detected parts (`smallparts.find_kind_parts` + `stamp_parts`, `pipeline._detect_parts`,
Balanced and Max, after the matte cut and before the backdrop decisions): grouping is colour
only, so the parts people personalise were grouped with whatever shares their colour (the
Ducati's yellow shock spring with the gold frame tubes; painting it painted the frame). Now:
- **Vocabulary.** Florence-2's `<CAPTION>` (`florence.caption`) picks the object class
  (`smallparts.object_class`: motorcycle, car, bicycle, sneaker, figure, generic: the class
  whose word comes first). Each class has a list of part `Kind`s (`smallparts.VOCAB`: key,
  label, plural, 1-2 bare-noun prompts, a size range per instance as a share of the object,
  an instance cap; tier `accessory`, used, or `panel` (tank, fairing, hood, roof), asked but
  never kept: panels split the one paint a user repaints in one click and cost region
  isolation when measured).
- **Detector.** OWLv2 large-ensemble (`partdetect.detect`, lazy singleton, fp16, local
  cache only, pinned snapshot) on the work image and its four 0.6 corner tiles, every phrase
  of the class in one pass per crop (the panel phrases compete for the boxes: a tank's box
  answers "fuel tank", not its runner-up "seat"). Florence-2's open-vocabulary detection
  (tile-sized boxes, confidences that do not separate right from wrong) and Grounding DINO
  (misses the small parts; its wheel boxes cost region isolation) were measured and rejected.
  `partdetect.detect_in` is the same model on given windows only (every phrase's score per
  box), for the wheel second look below. Like every Hugging Face model of the app (Florence-2,
  BiRefNet, ViTMatte), the pinned snapshot is looked up in the local cache first
  (`hfcache.snapshot_present`, `huggingface_hub.try_to_load_from_cache`, no network); a
  missing file makes the model unavailable before `from_pretrained` is called, which asks the
  network for its error message even with `local_files_only`.
- **Gates** (`PartGates`): OWLv2 score >= 0.3 (the sensitive knob: at 0.2 the wrongly named
  masks went from 5 to 33-61), per-kind box NMS (IoU 0.6 or 80 % containment), a box at most
  4x the kind's largest mask, at most 12 prompts per kind; SAM 2.1 box prompts on crops (box
  + 8 % + 8 px, `SamMasker.prompt_boxes`, chunks of 16 against `pipeline.PARTS_BUDGET_S` =
  6 s); a candidate is kept when unclipped, SAM's score >= 0.85, its box has IoU >= 0.45 with
  the detection box and fills >= 0.1 of it, >= 80 % of it lies on the matte, it fits the
  kind's size range; duplicates of one kind (IoU > 0.5 or 80 % inside) and of two kinds (IoU
  > 0.6) keep the best (score x votes x SAM); the kind's instance cap. A wheel box is split
  into tyre and rim by `wheels.split_wheel` (a wheel below 1.5 % of the object, or one that
  does not split, adds nothing); a wheel's SAM mask must fill 30 % of its box (a tyre ring
  alone fills 0.4: SAM's best-scored answer to the BMW's front wheel was a speckled 14 % mask,
  its whole-wheel answers scored a little lower), and the rim keeps only its pixels on the
  matte (the backdrop seen between the spokes: the BMW's white wall, the dark behind the
  Ducati's three spokes). A smaller part touching a detected mirror (3 px), at most half its
  size and inside its box grown by its size on each side is the mirror's stalk and is folded
  into it (`attach_parts`: OWLv2 called the Alpine's mirror stalk a "rear spoiler", a 633 px
  part row in the body's colour beside the 1925 px head; a grip next to a bar-end mirror is
  never folded).
- **Stamping** (`stamp_parts`): larger parts first, smaller over them; a region of IoU >= 0.6
  with the mask is adopted whole (tagged, pixels unchanged; a spring's or a mirror's region is
  often a better outline than SAM's box answer), a tyre or a rim adopting only a region lying 90 %
  inside its mask (the regions stage's own wheel split drew the BMW's front tyre region over the
  fork stanchion in front of it, 11.5k px beyond the tyre's mask, and adopted whole it made
  "Tyres" paint the fork); otherwise the mask is stamped
  pixel by pixel except over lettering, other parts, regions it only nicks (< half of them
  inside) and distinct sub-parts inside it (>= 85 % inside, <= 35 % of the mask, >= dE 15 from
  the rest: a caliper on a rim); a host region more than twice the mask is always cut; the
  remnant ring of a region the part cut (lost half, < 400 px left or less than 35 % of what it
  lost; for a tyre or a rim a remnant of 400 px or more only when too thin for a 3 px disk: a
  piece with a core stood in front of the wheel) joins the part when it is its colour (dE 15). A mask that is a union of regions (`PartMask.exact`: the wheel look's
  region and painted routes) adopts every region lying 85 % inside it within dE 10 of the
  adopted one too (the Corvette's caliper, seen above and below a spoke: adopting the larger
  piece alone left the lower one in a "Gold" colour group, a yellow blob under a red caliper);
  a SAM mask does not (a wheel's mask holds the fork leg in front of it). Every instance is one region of source `'kind'` with `part_kind`,
  `part_label`, `part_plural`, `part_instance` (0.. per kind, larger first), `part_score`;
  its backdrop decision is always object. The SAM masks stay in the analysis context for
  the junk pruning (rule 2 below) and are kept in the regroup seed.
- **The wheel second look** (`smallparts.find_calipers`, when the class has a `brake_caliper`
  kind: motorcycle, car, bicycle): no detector scores a caliper at the scale of the photo
  high enough to use (OWLv2 0.18-0.22 on the three reference bikes, below 0.12 on two cars
  with painted calipers), so every wheel mask that passed the gates (`select_parts(...,
  wheels_out=)`) is looked at again. The wheel's outline is the convex hull of its mask (a
  swingarm or an exhaust in front cuts notches into it, and the caliper sits in one; the
  ellipse inscribed in the box when SAM answered with a sliver of the tyre). Candidates:
  OWLv2 on the wheel's crop (+12 %), its boxes whose best phrase is the caliper (>= 0.1; the
  look's other phrases, disc, tyre, rim, hub, spoke, fork, swingarm, exhaust, sprocket, lug
  nut, valve stem, bolt, are there to take the boxes that are not one), and the regions of
  the partition (>= 60 px, >= 85 % inside the outline, compact: solidity >= 0.6, not
  lettering, not another part), either centred at 0.2-0.92 of the wheel's radius and 0.2-8 %
  of its area. A candidate is *verified* when OWLv2 on a square crop 2.5x its size names a box
  agreeing with it (IoU >= 0.4) a caliper at >= 0.22 (the bikes' calipers score 0.26-0.38
  there, the fork foot the BMW's caliper is bolted to 0.25, every other candidate of the set
  below 0.19); a box candidate gets its mask from a SAM box prompt and the gates above. A
  region is *painted* when it has a clear colour (chroma >= 20) whose paint (lightness-
  normalised (a, b) within 15) covers at most 5 % of the rest of the wheel and of a band
  around it (a quarter of the wheel's radius wide) and it is no piece of a same-painted region
  going on outside the wheel: a painted caliper is the one coloured thing inside a wheel, and
  OWLv2 does not recognise one at any scale (0.05-0.10), while the BMW's yellow fender edge
  inside its front wheel continues outside it. One caliper per wheel: the best piece
  (verified before painted) plus every accepted piece of that wheel within dE 12 of it (a
  caliper seen between two spokes). The calipers are stamped like every other part (kind
  `brake_caliper`, "Brake caliper" / "Brake calipers"). Found on the reference photos: 6 of
  the 7 visible calipers (the Ducati's gold one, the BMW's gold front and black rear
  verified; the Alpine's two blue and the Corvette's yellow one, both of its pieces, painted),
  none on the Torana, the Boxster, the van or the Jaguar; the bicycle's rim brake is not found
  (no wheel of its thin, motion-blurred rims passes the gates). The same pass looks for the
  brake disc (`smallparts.find_discs`, when the class has a `brake_disc` kind: motorcycle,
  bicycle) inside every wheel split into tyre and rim: the rim is everything inside its lip,
  so the Ducati's drilled steel disc was part of its gold "Wheel rim" and a rim repaint
  painted the disc. A box of the pass whose best phrase is the disc (>= 0.3), centred on the
  hub (0.3 of the radius) and 0.4-0.95 of the wheel's size is prompted as a SAM box; SAM's
  other answers to a disc box are the whole rim, so the disc is the answer (the gates' score
  and box agreement, on the matte) lying 85 % inside the rim, at most 0.7 of it and with at
  most a quarter of it on the rim's lip band (the outer 15 % of the rim), its holes and specks
  below 25 px closed (a drilled disc comes back speckled); stamped over the rim like every
  smaller part. Found: the Ducati's front disc (12.6k px); the BMW's front disc answer holds the
  fork foot and fails the gates, so its rim keeps the disc. The look costs 0.04-0.1 s on a car
  without visible calipers and about 0.5 s on a bike (up to six verification crops and one disc
  prompt per wheel).
- Measured on the ten-photo part reference set (194 must-be-separate parts, by-eye junk
  catalogue; two fresh analyses per variant): isolated parts 6 -> 18 (and 27 of the 133 parts
  in the vocabulary in a group of their kind only, 0 before), merged 185-186 -> 168-170, no
  loss of the segmentation ceiling (region isolation 0.490-0.495 -> 0.495-0.505 over the
  round's fresh runs, 0.505 in the final two); 51 masks stamped,
  40 on a reference part of their kind, 5 real parts under the wrong name (a rear brake disc
  as "Sprocket", a headlight as "Fog lamp", a side lamp as "Fuel cap", a whole Gundam head as
  "Head", a rim that holds its disc). The strict isolation counts a part alone in its group:
  since the second wheel of each bike is found (the Ducati's rear, the BMW's front), each
  bike's two tyres share one "Tyres" group until a Split, and the 20 isolated parts of the
  previous round are 18 (the kind-level count rises). evalkit: PACO unchanged (best IoU 0.348,
  R@.5 0.343, achievable recall 0.408); on our set region metrics up (best IoU 0.678 -> 0.682,
  R@.5 0.747 -> 0.753, achievable recall 0.820 -> 0.831, thin-part recall 0.474 -> 0.526),
  boundary F@2 0.790 -> 0.780 (the Ducati's disc outline). Found: shock springs, grips, seats, sprockets, exhausts, tyres, rims
  (both wheels of the two reference bikes), grilles, mirrors (with their stalks), bumpers, door
  handles, logos, bottles, pedals and (the second look) brake calipers and a brake disc; not
  found: levers, sneaker panels and laces, Gundam armour. About 0.4-1.2 s per image (caption
  0.06 s, OWLv2 + prompts + gates, the wheel look, stamping 0.02-0.08 s), +0.8 GB of VRAM while
  OWLv2 is loaded. On the twelve-photo set (the ten plus the Alpine and the Corvette, 221 parts)
  isolated parts 8 -> 26 of 221, every caliper found is in a group of calipers only, and the
  strict one-part-per-group count (the BMW's and the Alpine's two calipers share one group
  until it is split by instance) is 2 of 7 (1 of 7 without detected parts, the BMW's gold
  front caliper by its colour).
- Without OWLv2 (package or weights missing, a CUDA OOM, any model error) or without the
  caption (the generic vocabulary is used) the stage runs as before, logged; without the
  model the wheel look finds nothing.

The matte keeps one subject (`subject.other_objects` + `cut_off`, `pipeline._subject_matte`,
Balanced and Max, after the detected parts, for a caption of a car, a motorcycle or a bicycle):
BiRefNet's matte picks the salient object, and an object touching its silhouette from behind can
come with it. In the Torana showroom the red coupe parked behind the Torana had its door panel
inside the matte (0.95), so the backdrop decisions called it object, the clustering put it into
the Torana's orange "Rust" and the one-click repaint of the Torana painted half of the coupe navy.
SAM tells them apart where the partition cannot: prompted with the box of the matte's main
component it answers with the subject's silhouette (taken when it scores 0.9, lies 90 % on the
matte and covers 85 % of it; the most covering such answer), and prompted with a point at the
deepest pixel of a piece of the matte left outside that silhouette (grown by 5 px; the piece at
least 2 % of the subject and 1500 px, holding a 3 px disk) it answers with that piece's object:
when that object covers 80 % of the piece and lies at most 5 % inside the subject, the piece is
another object. A region lying half in such pieces goes with them whole (`whole_regions`: the
rest of the coupe's door region, the 5 px seam kept along the Torana and a patch of its roof,
stayed in the Torana's paint); the other object's matte is set to backdrop, the other regions are
cut along it whatever the shares (`cut_off`: the matte cut's 20 % rule would keep a panel in a
region it shares with the subject), and the backdrop decisions that follow give it to the
background. Detected parts and
their regions are never touched, and a pair of sneakers (two subjects of one product; SAM's box
answer on a pair covers one shoe) is not examined. Measured on the part reference photos: the
Torana's coupe (15.6k px, 4 % of the Torana: 48 px of the coupe's box change when the Torana is
painted navy, the whole panel before) and the red rope in front of the Corvette (10.5k px) go;
the BMW's licence plate (0.7 % of the bike), the Corvette's bumper corner behind a sign post (its
SAM object reaches into the car) and every crumb stay. 0.04-0.14 s (one image embedding for the
box, one for the points).

```python
# recolor/segmentation/hierarchy.py (the pipeline's regions stage runs it at Balanced and Max)
def find_pockets(labels, albedo_lab, max_pockets=24) -> list[dict]
def find_colour_pockets(labels, albedo_lab, max_pockets=16) -> list[dict]
def recover_parts(image_rgb_u8, albedo_lin, labels, info, prompter, progress=None) -> (labels, info, n_added)
# recolor/segmentation/sam_masks.py
SamMasker.prompt_parts(image_rgb_u8, points, crop=192) -> list[list[dict]]   # per point: SAM's 3 masks
```

6. Part recovery (see 7 for the matte cut that follows it): the automatic proposals at Balanced miss small parts of their own colour
   inside a large neutral mask (the yellow BMW's gold fork tube, ~700 px, inside the one
   mask of the black machinery: tinted blue by rule 8, its lit edge labelled paint; the
   Exia's red chin crystal, half in the face and half in the blue armour). A pocket is a
   2x2-opened blob of >= 60 px of albedo chroma >= 20 and ΔE >= 15 from its neutral host
   region (median chroma < 12, >= 4000 px); the 24 largest are prompted with one point each
   on a 192 px crop (SAM at its full input resolution on a small area, ~40 ms per point). A
   returned mask becomes a region (source 'prompt') when SAM's score is >= 0.7, it does not
   touch the crop border, covers half the pocket, has 150 px .. 40x the pocket, a median of
   chroma >= 18 and ΔE >= 15 from the host, and is not a region that already exists. A
   reflection of the paint in a chrome part does not pass: SAM returns the chrome part,
   whose median is neutral. Any failure keeps the partition (logged); Fast skips it.
   Colour pockets, the second kind: SAM paints a large mask's colour mode as one region even
   when it is scattered over the whole image (the yellow BMW's paint region also held the
   gold preload adjuster and red cap of the far fork top, which were painted navy with the
   paint). Every 8-connected piece of a chromatic region (median chroma >= 18, >= 2000 px)
   other than its largest, of >= 25 px and >= ΔE 8 from the region's median, is prompted
   too (the 16 largest). A part found there must be chromatic (>= 18), >= ΔE 12 from the
   host and >= 40 px (the other tests as above); it becomes a region of source `'part'`.
7. The matte (every preset): BiRefNet's foreground matte (`foreground.fg_prob`, 60-160 ms)
   cuts every non-exempt region with at least 20 % of its pixels on each side of the 0.5
   contour along it (`hierarchy.cut_on_matte`: the object side gets a new id, source
   `'split'` or the parent's `'part'` / `'prompt'`); a region mostly on the backdrop side
   also gives up every compact object piece the matte finds well inside the silhouette (an
   8-connected component of at least 150 px that holds a 7 px disk, one region per piece; a
   sliver along the silhouette, where the matte and SAM disagree by a few px, has no such
   core and stays). The rule was motivated by the yellow BMW's brake-fluid reservoir (1.4k
   px inside a 620k px backdrop region, which could never reach 20 % of it), but its
   translucent cup is backdrop to the matte as well (0 % of the cup body above the 0.5
   contour), so that one stays with the backdrop; measured, the rule is neutral on the six
   reference photos (+1 region per image) and +0.4 pt achievable recall / +0.6 pt f@2 on
   PACO. Then
   `grouping.backdrop_decisions`
   decides per region: the border set is the smallest set of groups of a plain clustering
   owning 90 % of the image border; a region with a matte share <= 0.15 in or next to that
   set is backdrop (`bg` 2), and so is one reached from there through a chain of regions of
   matte share <= 0.15 none of which is enclosed by the object (half of its boundary on regions
   of share >= 0.6): the pieces of the other cars in a showroom (the yellow car's sill, a lamp of
   the black car at the back, the letters of a poster) sat beyond the first ring and were
   decided object, tiny colour rows of the subject's panel; any other region of the set that the
   matte does not call object (share < 0.6) is backdrop too (`bg` 1), everything else is object
   (0), a see-through hole inside the object included (no chain reaches it). Without the model
   the border rule of the groups stage decides, as before. Measured on the six reference
   photos: scored-part pixels inside background groups 4.61 % -> 0.12 % on the product
   shots, backdrop coverage 0.59 -> 0.93, both halves of a split backdrop flagged.

```python
# recolor/segmentation/grouping.py
def group_regions(labels: np.ndarray, albedo_lin: np.ndarray, region_info: list[dict],
                  max_groups: int | None = None, delta_e: float = 10.0, photo_rgb_u8=None
                  ) -> tuple[list[Region], list[ColorGroup], np.ndarray]
# -> (regions, groups, group_map int32 HxW with values = group id in 0..G-1)
def regroup(regions: list[Region], labels, albedo_lin, max_groups, delta_e, photo_rgb_u8=None) -> same
def backdrop_decisions(labels, albedo_lin, region_info, fg, delta_e=10.0, max_groups=None) -> int8 [N]
def absorb_lit(regions, groups, labels, albedo_lab, photo_lab, params=DEFAULT_LIT) -> (regions, groups, group_map, log)
def merge_groups(groups, regions, group_map, labels, ids: list[int]) -> same
def split_group(groups, regions, group_map, labels, albedo_lin, gid: int, k: int = 2) -> same
def split_instances(groups, regions, group_map, labels, gid: int) -> same     # a part group into its instances
def move_regions(groups, regions, group_map, labels, region_ids: list[int], gid: int) -> same
def enforce_parts(regions, groups, labels) -> (regions, groups, group_map)   # every part region in its kind's group
def annotate_groups(groups, regions, group_map) -> groups                    # the panel view: minor, parent
```

Grouping: per-region median albedo in Lab (median, not mean — highlights and panel lines
skew means); area-weighted agglomerative clustering with CIEDE2000 linkage threshold
`delta_e`; if `max_groups` is set, keep merging the closest pair until the cap holds.
`ColorGroup.name = colornames.nearest_name(lab)` (a repeated name gets a suffix: "Copper",
"Copper 2"; the suffix does not make it a custom name), `hue_family =
colornames.hue_family(lab)`. Groups sorted by area descending, ids 0..G-1. Region ids stay
stable across regroup/merge/split/move (only `group_id` changes) except for split, which
appends new region ids.

Detected parts: a region with a `part_kind` (`Region.part_kind`, from the regions stage's
`'kind'` regions) never enters the colour clustering. Every kind is one group of its own,
whatever its colour and however small, outside the `max_groups` cap (like a locked part or a
decal), never background (the region's backdrop flag is forced off), unlocked and paintable. A
group is a *part group* when more than half of its area is one kind's part regions
(`ColorGroup.part`, with `part_label`, `part_plural` and `part_instances`, the number of
distinct instances), computed from the members on every rebuild, so the flag follows every
merge, split, move and regroup. Its automatic name is the kind's label, or its plural for
several instances ("Shock spring", "Wheel rims"); a custom name is kept (`_keep_name`
recognises the automatic part names). A part group never takes part in the lit merge
(`absorb_lit`: neither candidate nor anchor), is never the paint (`refine.main_paint`,
`paint_family`), never absorbs a washed-out region or a highlight and never loses one of its
regions to them (`refine.absorb_washed`, `materials.absorb_highlights`), is never locked as
another material (`lock_materials`), never takes in a decal or a carried-over region by colour
(`refine._nearest_group`), and the edge snap never moves its pixels (they are added to the
snap's protect mask). `enforce_parts` restores the invariant (every part region in its kind's
group, no region of another kind there) after the snap's relabel and after a regroup's
carry-over; it is a no-op in the normal case and puts back a piece a pixel-level split cut off
a part region. `split_instances` splits a part group with several instances into one group per
instance, no colour clustering (two same-coloured calipers split cleanly), named by where each
sits (`instance_names`: "Mirror (left)" / "(right)", "(upper)" / "(lower)", numbered "(1)",
"(2)" left to right beyond two; on a vehicle, a kind that comes in pairs along its length
(`AXLE_KINDS`: rims, tyres, calipers, discs, door handles, footpegs, seats, exhausts), with its
instances at least a fifth of the wheels' span apart, is "(front)" / "(rear)" when the parts
give the front away (`vehicle_front`: two wheel instances give the midpoint, and the grille,
the fork, the grips and the mirrors ahead of it and the sprocket, the spoiler, the exhaust and
the seat behind it vote), else numbered: in a side view left and right were the front and the
rear and read as the car's own sides; a car's twin exhaust tips, side by side, keep left /
right), each keeping the group's lock; a region of the group that is
not a part region of its kind (a shadow the pruning put into the part) goes with the instance
whose boxes it overlaps most. `merge_groups` of instances of one kind that still carry
automatic names gives the kind's name back ("Mirrors"). The instance names survive other
edits.

Panel view (`annotate_groups`, run by the pipeline whenever it writes a grouping, before the
protect mask is computed, and after a background toggle): a group that is not background, not
a part group and holds no lettering (sources `'text'`, `'named'`) is `minor` when it covers
less than `MINOR_FRAC` (0.4 %) of the object (the area of the non-background groups) *and* is
its `parent`'s colour under other light: within `MINOR_DE` (12, CIEDE2000) of it, or, both
with a body colour (chroma >= `MINOR_CAST_MIN_C`, 8), of the same body colour
(lightness-normalised (a, b) within `MINOR_CAST`, 12: a shadow is darker but keeps the paint's
body colour; two neutrals always pass that test, and a paper card in a window, L 62, was "minor
next to" a charcoal L 37 at dE 27, a white badge on black trim would have been). Its `parent` is the non-background, non-minor group owning
most of a 5 px ring around it (-1 when none). Size alone made the Minor section a drawer of
every small group: a tiny group of a colour of its own is a real small part (the Ducati's
gold preload adjuster, 690 px next to silver at dE 25; the Torana's amber tail light, dE 15
from the paint) and stays among the colours. A view hint only: minor groups are grouped,
painted and edited like any other. A minor group whose parent is a part group (the part under
other light) is never in the paint family (`refine.paint_family`): painted with the paint,
the robot's red feet got a navy rim. The rule is versioned (`grouping.PANEL_RULE`, 3: 1 size
alone, 2 the lighting-variant test, 3 its body-colour part only between two colours) and stored
with the groups (`panel_rule` in `job.json`); a job whose groups carry the view of an older
rule is annotated again in memory when it is served (`pipeline.refresh_panel_view`, from
`GET /api/jobs/{id}` and the event stream; skipped while an edit of the job runs), so a job
analysed when size alone made a group minor no longer hides a real small part under the
collapsed divider (the Ducati's 690 px gold preload adjuster; the round-6 Corvette's windscreen
reflection "Silver 2", minor next to "Charcoal" by the neutral cast test, is a colour row
again). `job.json` is not rewritten on open; the next edit or state save stores the new view.

Background: with the regions stage's matte decisions (`info[i]["bg"]`, `Region.backdrop`)
the backdrop regions and the object regions are clustered separately (`max_groups` caps the
clustering: the object side is clustered first with `max_groups - 1`, the backdrop takes what
is left, at least one group; `max_groups` 1 gives one group, which is never flagged
background since flagged it would leave nothing paintable while the background is ignored,
and `_mark_background` keeps that rule through every rebuild: a part group is never
background, and neither is the only colour group left (a regroup at 1 left one colour group,
which a rebuild in the refinement then flagged background: nothing to paint but the parts);
the refinement below then adds a locked group per isolated part and a group per decal island
that finds no group within ΔE 10, so a regroup of the yellow BMW at 3 shows five rows: three
clusters plus two locked parts, and the toast gives the real count), so no group mixes them
and every backdrop group is flagged
`is_background` (more than one may be); a region flagged only through the border set that
is not connected to the image border through flagged regions leaves the background and
joins the nearest unflagged group within dE 10, else one of its own kind (a see-through
hole the matte called backdrop keeps its flag). Afterwards a group is background when at
least half of its area is backdrop regions (`_mark_background`), so the flags survive every
rebuild in refine and every regroup. Without a matte the one group owning more than 0.35
of the border is flagged, as before.

One paint under different light (`absorb_lit`, with the photo): the albedo keeps part of
the shading, so a paint clustered into a lit and a shadowed group (the RX-78's chest sides,
the BMW's panel under the mirror). After the clustering every chromatic group (normalised
chroma >= 12, raw chroma >= 18, not background, not locked) joins the closest larger anchor
(a non-background, unlocked group of chroma >= 30 and >= 2000 px) whose lightness-normalised
albedo mean is within CIEDE2000 (kC = 2) 8 and whose lightness-normalised photo colour is
within 7, unless the material veto fires (the candidate is duller than 0.6 of the anchor's
chroma and more than 10 L darker: a gold caliper, a stalk tinted by the paint's bounce).
Every test is against the anchor's own colour, never a drifting centroid, and chains
resolve to their final anchor; neutral pairs are never touched. The intrinsic shading layer
does not carry this difference (measured: the decomposition put the whole lightness gap of
the RX-78's shadowed chest into the albedo), the photo's lightness-normalised chromaticity
does. Measured on the six fixture jobs' stored labels: the main paint's share of its family
BMW 0.927 -> 0.942 (paint groups 2 -> 1), Sazabi 0.954 -> 0.981, RX-78 0.531 -> 0.967, red
car 0.858 -> 0.988, Ducati and Exia unchanged, zero known wrong merges.

The sheen merge (`refine.absorb_sheen`, `grouping.SHEEN_LIT`: `absorb_lit` with `sheen_only`),
the last step of the refinement and of a refined regroup: a chromatic group whose photo is
lighter than an anchor's and duller in the same hue (lightness-normalised hue within
`sheen_hue` 12 deg, at least `sheen_l` 6 L lighter) and whose albedo is within the lit merge's
tolerance joins it: the paint under a glossy sheen, which the photo test above rejects
because the sheen washes its colour out (the Torana's roof and boot lid, "Brick" beside the
"Rust" body, a separate group a one-click repaint left orange). Run in the clustering, the
merge regressed the Torana's flare shadow (the snap then shrank it from 222 to 191 px and the
pruning's shading test no longer explained it), so it runs after the snap and the pruning.

```python
# recolor/segmentation/refine.py  (called by the pipeline's groups stage)
def refine_groups(photo_u8, albedo_lin, labels, regions, groups, group_map, snap, progress=None,
                  parts=None, residual=None, shading=None, fg=None, part_masks=(), prune=None) -> Refined
# Refined: labels, regions, groups, group_map, islands (bool HxW), protect (bool HxW), report,
#          origin (int32 [N]: the input region each final region descends from, -1 for none)
def regroup_refined(photo_u8, albedo_lin, labels_input, origin, labels, regions, islands,
                    max_groups=None, delta_e=10.0, parts=(), bg=None, residual=None, sources=None,
                    part_tags=None, shading=None, fg=None, part_masks=(), prune=None,
                    user_flags=None) -> (regions, groups, group_map, protect)
# recolor/segmentation/junk.py  (a step of refine_groups and regroup_refined, with `prune`)
def prune_junk(photo_u8, albedo_lin, shading_lin, labels, regions, groups, group_map, islands, fg=None,
               part_masks=(), params=DEFAULT, user_flags=None, diagnostics=False)
    -> Pruned   # regions, groups, group_map, islands, protect, log (+ kept, to_part, guarded with diagnostics)
def shadow_test(pair, params) / gradient_test(pair, params) / significance_test(share, pairs, params)
def rim_test(alb_group, alb_part, params)                 # a part's rim: the part's own colour, a shade darker
def refine_after_edit(photo_u8, albedo_lin, labels, regions, groups, group_map, islands, regrouped)
    -> (regions, groups, group_map, protect)
def fill_decal_gaps(labels, albedo_lab, regions, groups) -> (labels, regions, groups, group_map, moved_px)
def fill_letter_counters(labels, albedo_lab, regions, groups) -> (labels, regions, groups, group_map, moved_px)
def absorb_sheen(photo_u8, albedo_lab, labels, regions, groups, group_map, islands, protect)
    -> (regions, groups, group_map, protect, moves)       # the refinement's last step
# recolor/segmentation/smallparts.py (the regions stage's detected parts)
def find_kind_parts(image_u8, albedo_lin, fg, caption, detect, prompter, budget_s=..., labels=None,
                    info=None, zoom=None, look=DEFAULT_WHEEL_LOOK) -> (parts, report)
def find_calipers(image_u8, albedo_lab, fg, labels, info, wheels, zoom, prompter, kind, object_px, ...,
                  pass_out=None)                          # pass_out gets the zoomed pass for find_discs
def find_discs(image_u8, fg, wheels, wheel_pass, prompter, kind, object_px, ...) -> parts   # the disc inside a split wheel
# recolor/segmentation/grouping.py (instance names)
def vehicle_front(regions) -> -1 | 0 | 1                  # which side a vehicle's front is on, from its parts
def instance_names(label, centroids, front=0, axle=False) -> list[str]
# recolor/segmentation/partdetect.py / hfcache.py
def detect_in(image_u8, crops, phrases, min_score=MIN_SCORE) -> list[dict] | None   # boxes with every phrase's score
def snapshot_present(model_id, revision, files=("config.json",)) -> bool          # local cache only
# recolor/segmentation/materials.py
def shine_features(labels, albedo_lin, photo_u8, residual=None) -> ShineFeatures
def tag_regions(regions, feats) -> regions        # Region.shiny, Region.chrome (advisory)
def absorb_highlights(regions, groups, labels, feats) -> (regions, groups, group_map, moves)
def chrome_regions(regions, feats) -> set[int]
# recolor/segmentation/matting.py
def snap_labels(image_u8, labels, group_map, groups, protect=None, progress=None) -> (labels, method)
def status() -> str; def is_loaded() -> bool; def warmup() -> None; def release() -> None
```

Refinement, in order (every step keeps a complete int32 partition, ids 0..N-1). "The
paint" is the largest non-background, unlocked colour group above CIELAB chroma 18 plus
every unlocked chromatic colour group within 25 deg of its hue (a part group is never in
it, nor a minor group whose parent is a part group, and steps 1-4 leave the part groups and
their regions alone, see Detected parts above).
1. Absorb: a region touching a clearly chromatic group (C >= 30), within 12 deg of its hue,
   0.5-0.95 of its chroma, >= 3 L lighter, not larger, with chroma falling as lightness
   rises (corr(L, C) <= -0.2), joins it: paint washed out by a highlight. Then a part of
   source `'part'` that clustering put into the paint (less the groups step 4 will lock) is
   another material in a colour close to the paint's (a gold fork tube between the BMW's
   fairing and fender, a red sticker on the Ducati's shock, a red car's tan console): it
   gets a group of its own (parts within ΔE 10 share one), locked (`refine.isolate_parts`).
   Then the highlights (`materials.absorb_highlights`): a region whose albedo the highlight
   pushed toward white (albedo L >= 55, chroma below 0.95 of the paint's, photo median L >=
   50, at least 15 % highlight pixels: clipped, glints or a positive residual above 30 % of
   the pixel's brightness) joins a touching paint (>= 20 shared px) when its unclipped
   remainder (the unclipped pixels under the 60th luminance percentile, 2 px inside the rim)
   carries that paint's hue (within 18 deg, normalised chroma >= 10, hue stability >= 0.6).
   Guards: the candidate is not backdrop, not lettering or a small distinct part, at most
   half the paint's area, and not a lamp (more than 60 % clipped). The same features give
   every region its shininess (`Region.shiny`) and a chrome advisory (`Region.chrome`: a
   >= 500 px region that glints or clips on >= 25 % of its pixels, near-neutral albedo of
   L 25-78, a remainder of L <= 70, no stable hue or few chromatic pixels, a wide luminance
   spread); the group carries the area-weighted `shiny`, the glint share `glint`
   (`Region.glint`: clipped or specular pixels) and a `finish` badge ('shiny' when the
   glint share is >= 20 %, 'chrome' when at least half its area is chrome-tagged) for the
   UI; the engine reads only 'chrome' (such a group is never a neutral source, section 3.5,
   and measures a group's glint share on the layers itself). The badge follows the glints, not `shiny`: a strong positive residual is
   near-universal on glossy paint (98 of the yellow BMW's 113 regions at 15 %), glints at
   20 % mark two to five groups per photo (the chrome and the highlight zones). Nothing
   is locked automatically from these (chrome recall was 6 of 29 in the round-3 measurement,
   with false hits on dark glossy plastic).
2. Decals: in the main paint's regions, 8-connected blobs >= 200 px that are > 25 dE from
   the region median, > 30 lower in chroma and a bright neutral (L >= paint L + 10, C <= 12)
   become their own regions (nearest group within dE 10, else their own group). So does a
   neutral blob of >= 40 px only 3 L lighter than the paint when >= 10 px of its outer
   ring touch a neutral group within dE 6 of its colour: a clear or white part continuing
   into its neighbour (the BMW's tail-light lens inside the tail's SAM mask, which turned
   grey with a dark contour). Their pixels are the `islands` mask; the letters'
   anti-aliased rim stays with the paint. Regions the regions stage stamped as lettering or
   small distinct parts (`'text'`, `'small'`) are islands too: the snap never claims them
   and the paint's ramp never enters them. Then the decal gaps (`refine.fill_decal_gaps`):
   in a region outside the paint of at most 4000 px touching a main-paint region by >= 20
   px, whose median is > 20 dE from that region's, blobs of at most 300 px within dE 10 of
   the paint region's median and > 20 from the region's own go back to the paint region
   (the yellow between the letters of the BMW's "RR S1000", which SAM's mask of the decal
   held and a navy repaint left yellow). Then the letter counters
   (`refine.fill_letter_counters`): the paint seen through a letter (the holes of an "8", the
   triangle of a "4") is the paint. Inside a lettering region (source `'text'`), or inside a
   small distinct region (`'small'`) at least 40 % of whose 3 px outer ring is lettering, every
   8-connected blob of 4-300 px in the main paint's hue (within 22 deg) with at least 0.6 of
   its chroma, paint-like in its median too, goes to the main-paint region touching the
   lettering most, pixel by pixel (a letter's outline stays with the letter). Only lettering
   printed on the paint counts: at least half of its 3 px outer ring is the main paint and its
   own median is not paint-like (else the strokes pass as counters: the navy letters of a FILA
   logo on blue jeans). The Ducati's "748": its counters are the red paint mixed with the white
   and gold of the letter edges (lighter, 15-20 deg toward orange, 0.8 of the chroma), so
   neither the decal-gap rule (within dE 10) nor the pruning (islands are exempt) took them and
   a navy repaint of the paint left them red; now 189 px go back (4 on the Alpine, 22 on the
   bicycle).
3. Edge snap: each chromatic group (C >= 18, not background) is matted with ViTMatte-small
   in a 6 px trimap band; a band pixel with alpha >= 0.5 joins the group (the label of its
   nearest region). Pixels only move *into* chromatic groups, from whichever groups sit in
   the band (a neutral part, or another chromatic group, where the higher alpha wins);
   neutral groups never grow and islands are never claimed. Without `transformers` or the
   weights: every label snapped to the photo with a colour guided filter (radius 2),
   islands kept, and one logged warning; a CUDA OOM on the shared card, or any other model
   error on an unusual image (a panorama 3 px tall), uses that fallback for the one job
   and retries ViTMatte on the next. The processor is told the channel axis
   (`input_data_format="channels_last"`). The weights are read from the local Hugging Face
   cache only (`setup.sh` downloads the pinned snapshot): a download inside the groups stage
   held the GPU lock for the length of the network timeouts on an offline machine. Missing
   weights are looked for again on the next job.
4. Material lock: a paint-family group below 0.6 of the main paint's chroma and more than
   10 L darker is another material (a gold caliper, a fork cap) and is locked. A group more
   than half of whose area is recovered parts stays locked through every rebuild (a decal
   or a split piece joining it no longer unlocks it).
5. Protect: a region outside the (unlocked) paint and the islands, >= 100 px, whose photo
   pixels are >= 40 % old-paint-hued (C > 20, within 30 deg) is an object of that colour
   (a brake-fluid reservoir): the `protect` mask, which the engine's rule 8 never enters.
6. Junk pruning (`junk.prune_junk`, with `prune`; the pipeline passes `junk.JunkParams`): a
   tiny group that is only a lighting or colour-cast variant of a touching group (the red
   coupe's flare shadow, a dark-red strip the clustering gave a group of its own and step 4
   locked, so a repaint of the body left it red) joins that group. Candidates: the regions of
   every non-background group below 0.4 % of the object (the matte), smallest group first,
   each merge visible to the next; every comparison is the region's pixels against the
   neighbour's pixels in a 5 px ring around it (at the seam, not against a median). Four
   tests: *part rim* (rule 2b: a region lying (90 %) within 4 px of a detected part's SAM
   mask, whose object ring the part owns at least half of, and whose albedo is the part's in
   shadow, `junk.rim_test`: no lighter and no more chromatic than the part (+5), and for a
   coloured part still in its hue (within 30 deg, chroma >= 8): SAM's box-prompted mask stops
   a few px short of the outline and the band between is a superpixel of its own; the robot's
   red feet kept a dark-red rim of 900-1300 px as a group, which the material lock locked and
   the shadow test could not decide, and a navy repaint of the paint the rim clustered with
   drew a seam around the red feet; the rim is decided before the matte exemption, since the
   rim sits on the silhouette); *shadow* (the neighbour owns >= 30 % of the object ring, the
   photo is darker (ratio <= 0.85; only darker candidates are tested: a lighter branch took a
   rider's glove for a highlight), the intrinsic shading layer carries >= 30 % of that
   log-luminance step, and the chromaticity moved like light: duller with the hue kept (35
   deg), or the photo's chromatic log-shift equals the shading's (0.08); against a chromatic
   neighbour (>= 20) a candidate that kept less than 0.6 of its chroma is undecidable and
   kept); *gradient* (no photo edge on the shared boundary: median |grad log Y| and |grad log
   chromaticity| <= 0.12, albedo cast <= 8, photo dE <= 6); *significance* (the group below
   0.4 % and not locked, the closest-coloured touching group of the same body colour:
   lightness-normalised (a, b) within 12, photo dE <= 6). The part rim and the shadow test
   move single regions; the colour tests only dissolve a whole group. Rule 4, *crumbs*: a
   group of at most 150 px whose pixels are only crumbs (at least 4 8-connected pieces, none
   above 40 px) joins the group owning most of its ring, even a small distinct part's island
   (the 58 px left of the Ducati's gold "748" edging in 18 pieces, a "0.00 %" row that kept an
   orange fleck on the "8" under a navy repaint); lettering and detected parts never. Exempt from
   the other tests: lettering, named parts, the wheel split, recovered parts (sources `'text'`,
   `'named'`, `'wheel'`, `'part'`), groups at least half decal islands, groups mostly off the
   matte, and every part group (rule 1: a detected grip or door handle is tiny and has its
   neighbour's colour on purpose; without the rule the tests dissolved 4 of 35 detected parts).
   A region the user voted on (`user_flags`, from `user_flags.json` on a regroup: a lock or
   background choice, recorded for every region of the group it was set on) moves only into a
   host that ends with its vote (`junk.vote_agrees`: the area-weighted majority of the host's
   votes, as the regroup re-applies them, else the host's own flag). A sliver the user split off
   and locked alone keeps its group; a pruned sliver of a group the user locked carries that
   group's vote and goes back into it: exempted outright, it came back as a locked junk group
   after every lock and regroup (the robot's 1311 px feet rim, the BMW's 312 px "Army green",
   the Alpine's 1035 px "Graphite 5", three backdrop crumbs of the Ducati's un-flagged
   backdrop). Locks: a group the material lock locked
   is a candidate for tests 0-2 (evidence that the pixels are one surface under different
   light), never for the significance test (a colour argument, which the lock's own colour
   argument outranks). Nothing merges into a background group or a smaller one, and a part
   group hosts only a candidate that lies at least half under that part's SAM mask (dilated
   2 px) or is its rim (rule 2: such a shadow goes to the part even when a test picked another
   neighbour, provided the part is the larger of the two). With `backdrop_crumbs` (the
   pipeline turns it on when the fresh clustering is a product shot, i.e. the background is
   ignored by default) a tiny background group joins its closest-coloured background
   neighbour. The label map never changes; moved regions leave the islands and the protect
   mask is recomputed. `diagnostics=True` also returns the kept candidates with their reasons
   (`kept`, `to_part`, `guarded`; off in the product). The switches `JunkParams.part_rim`,
   `shadow`, `gradient` and `significance`, `prune_junk(diagnostics=)`, the cross-detector
   votes of `PartGates` and `LitParams(sheen=, sheen_only=)` beyond `SHEEN_LIT` are kept on
   purpose: the unit tests switch single rules off to test each alone and the experiment
   harnesses read the diagnostics; the product runs every rule. A seed records the pruning's
   parameters, and a field it does not have (a rule added since, `junk.LEGACY_OFF`: the
   crumbs) reads as that rule's off value, so a regroup of an older job reruns the pruning its
   analysis ran. Measured on the ten-photo reference set (fresh analyses): by-eye junk
   groups 16 -> 9-10, their severity 25 -> 13-15, the robot's feet rim folded into the feet (2-3 rim
   moves); counted as lost by the strict test: the BMW's far gold preload adjuster, which
   shares a locked "Bronze" group with its twin on the near fork, the 748 decal's drop-shadow
   crumbs (rule 4), and the paper card in the window of the coupe behind the Torana, which is
   background with the coupe now. About 0.3 s per photo on the CPU. What stays needs other
   signals: reflections of other objects are in the albedo, cavity shadows stamped as small
   distinct parts are islands on purpose.
7. The sheen merge (`refine.absorb_sheen`, see "The sheen merge" above): after the
   pruning, the paint under a glossy sheen joins the paint. Measured: the Torana's roof and
   boot lid ("Brick", about 18k px) are in the body's group, so the one-click repaint of the
   body paints them; the flare shadow stays pruned.

If refinement fails on an unusual photo (anything but a CUDA OOM, which the stage retries),
the plain clustering is kept, the job is treated as unrefined (no masks, no seed) and the
stage message says the edge refinement was skipped. The groups message gives the final
region count, which the summary tile shows, and how it differs from the regions stage's
("73 regions (72 + 1 carved out)").

Group edits keep a refined job consistent: the islands are pixel facts and never change,
the protect mask is recomputed after every edit (and after a lock or background change;
CPU only, outside `gpu_lock`), a pixel-level split rewrites the label map, a split's
halves inherit the parent's lock and background flags (a background group's halves are
both background), and a piece a pixel-level split cuts off lettering, a named, recovered,
small distinct or detected part keeps its parent's source (and with it the pruning's
exemptions and the island treatment; as source `'split'` a cut letter lost them). A regroup
reproduces the analysis (`refine.regroup_refined`): the groups stage records, for every
final region, the region of the regions stage it descends from (`Refined.origin`, stored
with that label map in `regroup.npz`), and a regroup clusters those pre-snap regions again
with the absorb rule, carries the result over (a decal island or a piece cut off by a
pixel-level split joins the nearest group within ΔE 10) and re-applies the lock and the
protect mask; the seed also lists the `'part'` regions, so step 1's isolation is redone,
every input region's backdrop decision (`bg`), so the backdrop is clustered apart again,
every input region's source, so the highlight absorb's decal guard sees the lettering
and the small distinct parts again, and every input region's detected-part tag, so every part
kind gets its group back (at any `max_groups`); the photo's lit / shadowed merge and the
highlight absorb run again too, and for an analysis that ran the junk pruning, so does the
pruning, with the parameters, the object mask (the matte > 0.5) and the parts' SAM masks the
seed recorded (a seed without them, from an older analysis, regroups as that analysis did,
with no parts and no pruning), and the sheen merge last. With the
job's own options it gives back the analysis's groups exactly; clustering the snapped
regions instead split the BMW's tan undertray off the paint. A regroup then gives back the
user's own lock and background choices: `user_flags.json` records them per region when they
are set, the pruning moves a voted region only into a host that ends with its vote (`user_flags`,
step 6: it runs before the flags are applied, and dissolved into a group of another flag the
choice would lose its majority), and a new
group more than half of whose area the user flagged takes the area-weighted majority of
those choices (a bracket locked by hand came back unlocked while its mapping was carried
over); a tie keeps the automatic flag. A custom name survives too (`pipeline.carry_names`):
a new group takes the custom name of an old group when more than half of the old group went
into it and it makes more than half of the new group (an automatic colour or part name is
never carried: the new group gets its own). Jobs analysed before these
files existed keep the older behaviour (a regroup of the current regions plus steps 1 and 4;
no masks at all for jobs older than refinement).

A part split by instance (`split_instances`) gives every instance the part's albedo as its
reference (`ColorGroup.ref_lab`), so the engine paints every instance from one source colour
and the split alone changes no pixel; the reference is dropped when an edit changes the
group's regions (a merge, a move) and carried by the rest (a rename, a lock).

"Ignore background" (`ignore_background` in the job record, set by the analysis and through
`PUT /state`): the renderer, the exports and the mapping suggestions see every background
group as locked (`pipeline.effective_groups`), so the backdrop is never repainted, never
suggested and never touched by the reflection stage (rule 7 treats locked groups as an
object of their own colour, which is what the warm-backdrop protection did by measurement
before); the stored flags are untouched, so the user's own locks and the regroup work as
before, and any group can be marked or unmarked as background from its row. With the
setting off, background groups are ordinary groups (suggested, paintable). The analysis
starts it on, unless the background is fragmented like a scene: more than two thirds of
the groups are background, or at least six background groups cover more than 75 % of the
image (`pipeline.default_ignore_background`). The matte picks one salient object, so in a
street scene or a showroom every other subject is flagged too, and a studio that opens with
20 of 29 rows locked is no use; the share alone is not a scene (a pair of sneakers on a
white backdrop is 78 % background in two groups and stays ignored). Measured: the product
shots stay on, their backdrop covering 0.55-0.78 of the image in 1-21 of 7-41 groups; the
street scene opens off at 0.83 and 20-22 of 29-31 groups (run to run; both rules), and so
does a model kit in a diorama city (0.86 of the image in 19 of 38 groups: the second rule;
its buildings, tanks and road are paintable like the street's taxi, and one click ignores
them), and the groups message says "background kept paintable". A change of the
setting is applied like a group edit (`pipeline.save_state`: the job's edit lock, an edit
generation, the cached renderer's flags updated in place), so a renderer being built during
the switch is rebuilt instead of being cached with the old locks.

User parts (Select part, Find part): the user points at a part the automatic pass missed (the
Ducati's far-side front caliper, seen between the spokes, sat in its "Brake disc" group) and SAM 2
cuts it out as a group of its own.

```python
# recolor/segmentation/sam_masks.py
SamMasker.prompt_session(key, stamp, load_image) -> (PromptSession, computed)  # LRU of PROMPT_SESSIONS (3) jobs
SamMasker.forget_prompts(key=None); SamMasker.release()                        # release() drops every session too
PromptSession.predict(point_coords, point_labels, box, mask_input, multimask, crop=None) -> (masks, scores, low, computed)
# recolor/segmentation/interactive.py   (no torch: any object with predict + shape does in tests)
def parse_prompt(body, width, height) -> Prompt        # PromptError on bad points / labels / box / pick
def segment(session, prompt, refine=True) -> Answer    # mask, alternatives, pick, crop, steps, timings
def answer_json(answer, width, height) -> dict         # 1-bit PNG data URL of the bbox, bbox, area, score
# recolor/segmentation/userparts.py     (numpy, CPU)
def absorb_parts(mask, labels, regions, share=ABSORB_SHARE) -> (mask, taken)   # part instances covered 90 %: whole
def carve(labels, mask, min_px=MIN_PIECE_PX) -> Carve | None
def keep_names(old_groups, groups, new_of, recompute=(), restore=None) -> None # a user-part edit renames nothing else
def normalize(regions, groups, labels) -> (regions, groups, group_map, changed)
def apply_registry(regions, groups, labels, registry) -> (regions, groups, group_map, changed)
def sync_seed_tags(tags, origin, regions, n_in) -> (tags, changed)
```

- **Prompts.** Points `[x, y, 1|0]` in work pixels (click order; a label is the integer 1 or 0, a
  coordinate a finite number: an integer too large for a float, or a body nested too deeply for the
  JSON parser, is a 400 like any bad prompt; NaN, Infinity and 1e400 are no JSON numbers, a 400 on
  every route; a JSON body above 1 MB is a 413, read in chunks and given up past the limit whether or
  not it has a Content-Length) and an optional box.
  The prompt is replayed the way SAM's demo refines a mask, without state on the server: step one is the box
  with the first positive point (three candidates with `multimask`, `pick` chooses the one the
  chain continues from: by default the best score, for a box the best score x box agreement), each
  further point is one step fed the previous step's low-res logits. Measured on the far caliper:
  two positive clicks in one single-output call answered with the whole wheel (85k px), replayed
  they give the caliper (1.4k px; 2.2k px with a third click). A part whose extent is at most
  `REFINE_MAX_SIDE` (0.45) of the long side is replayed again on a crop (its box grown by 0.35, at
  least 160 px, snapped outward to a 16 px grid so the next click reuses the crop's embedding): SAM's
  256 px logits are 6 px wide at 1536 px. The refined mask is taken when the crop does not clip it
  and it overlaps the full-image answer by IoU 0.5; the crop goes back with the answer and a commit
  that sends it back replays the same steps. Pieces and holes under `MIN_PIECE_PX` (12) are cleaned.
- **Embeddings.** One `PromptSession` per job (the work image's embedding plus the last crop's) on
  the shared model, keyed by the work image's file stamp, dropped with the model by the idle unload
  (`SamMasker.release`) and by `DELETE /jobs/{id}` (`pipeline.invalidate`, before and after the job's
  directory goes; a prompt that cached a session meanwhile finds the job's `deleted` flag and drops it
  itself). A prompt waits for the GPU lock at most `PROMPT_GPU_WAIT_S` (1.5 s; an analysis stage or a
  full-resolution export holds it) and then answers 503 with `Retry-After` `PROMPT_RETRY_AFTER_S` (2);
  while it waits it checks every 0.1 s whether its client went away (the studio aborts a superseded
  click) and gives up without running a model (`PromptCancelled`): 45 aborted prompts during a SAM
  stage left a health request answered in 31 ms (40 threads would each have been held 1.5 s).
  `Request.is_disconnected` never saw a disconnect behind the app's HTTP middleware, so the check
  listens for the client's goodbye for 5 ms (`app._client_gone`). It counts as activity for the idle
  unload. Measured (5090, idle, the Ducati's 1536 x 1024): the embedding 0.12 s after a load,
  0.023-0.029 s warm, a prompt 3 ms, a crop's embedding 0.022 s. On the shared card: `/segment` round
  trips 20-35 ms warm for one to three clicks (65-70 ms from the click to the outline in the
  browser), 51-69 ms for the first click on a part (its crop embedded; the cleaning of pieces and
  holes works on the mask's box), 1.1-2.2 s when SAM had been unloaded (the studio prepares the
  embedding when the tool opens: 14-47 ms with the model loaded).
- **The carve** (`groups/from_mask`: the prompt runs again on the server, a mask from the client
  is never used; it runs under the GPU lock alone, before the job's edit lock is taken, so a lock
  toggle or a rename of the job never waits behind a commit that waits for the GPU, and the carve,
  the stats and the writes run outside the GPU lock). `take` (the pill's "Take all", at most 8 group
  ids of groups under the selection) adds those groups' pixels to the mask. A part instance (every
  region of one `(part_kind, part_instance)`, measured together: a two-region user part covered
  99.5 % left its 225 px region behind as the old part) the mask covers at least `ABSORB_SHARE` (0.9)
  of is taken in whole (`userparts.absorb_parts`: Find's "spring" over the detected "Shock spring"
  left a 0.02 % "Shock spring" group, and selecting the far caliper again cut the earlier "Far
  caliper" part to a 112 px sliver); a far instance of the same kind is never taken. A group taken in
  whole, a part group or a colour group (the red tank selected exactly, the Alpine's small "Blue"
  badge group), that makes up at least `REPLACE_SHARE` (half) of the new part is *replaced*: the new
  part keeps its paint and, when the user gave no name, its name (selecting a part again gives it a
  new outline; a colour group's paint was lost before). The answer says what the part took in
  (`created_part.took_in`, `replaced`), and so does the tool's pill before Enter. `userparts.carve`
  keeps every region id: a region wholly inside the part keeps its id and joins it (all of it, its
  own isolated specks included: a 1 px piece of a group told to join stayed behind as that group),
  the pieces cut off the other regions become one new region (the next id, source `'user'`), other
  pieces of the mask under 12 px merge back into their region, a speck a cut region would keep next
  to the part joins the part, and larger holes stay with the regions they are in. The part's regions
  are tagged like a detected part (`part_kind` `user_<n>`, the label the name the user gave, the
  replaced group's, or "Part <n>"), so every part-group rule protects them: never clustered by colour,
  one group whatever the colour and size, never flagged background by the automatic rules, never the
  paint, never locked as another material, never pruned as junk. Their stats (area, box, median
  albedo) and shine cues (on a window around the part) are measured again, the protect mask is
  recomputed (the islands are pixel facts and stay), and every file is written atomically inside
  the job's edit lock and generation. No other group changes its name (`userparts.keep_names`: the
  rebuild renumbers repeated colour names by id, so cutting a part out of "Silver 2" renamed the
  untouched "Silver 3", and the donor "Gold" became "Yellow"; a part group's automatic kind name
  follows its instance count: "Brake calipers" that lost one is "Brake caliper"), its lock or
  background flag (a detected part flagged background by hand lost the flag on an unrelated commit)
  or its paint (the mapping is re-keyed by group id, not carried by membership); the new part is
  unpainted unless it replaces a group. The regroup seed gets the same carve (the part is an input
  region of its own, tagged, and the new region descends from it), and `user_flags.json` lists the
  part (`parts`: label, regions, `donors`, `was`: the tags of the regions it covered, `groups`: every
  group it took in whole with its regions, name, flags and paint, `homes`: for a region of a group it
  took only in part, a region of that group left outside, and `new_id`, the region its carve made;
  `parts/<kind>.npz` keeps where each of that region's pixels came from), which `apply_registry` uses
  as the safety net after a regroup. A user part the new one takes in whole loses its regions in the
  registry and waits there inside it (`inside`: the new part's kind; `_sync_user_parts` keeps it while
  that part is there); a user part that loses some regions lists them no more. Measured on fresh jobs
  (commit round trip, server split): 0.11-0.12 s for a part of 3-4k px after the first (SAM 10-14 ms,
  the carve 7-10, the stats 6-8, the shine cues 6, the groups 6-8, the protect mask 15-18, the seed
  8-11, the writes 45-50); 0.18 s for the first commit on a job with the tool open (a 90k px part, its
  shine cues 32 ms), 0.22 s without the tool's warm-up (the layers and the glint reference loaded then);
  0.33-0.40 s for a third of the image (the Corvette's body: the shine cues' window 170-180 ms). The
  writes are the label map and its id PNG (13-15 ms), the seed (7-10, zlib level 1: `savez_compressed`
  took 24 ms), the grouping (17-23); the Regions, Groups and edges display layers are drawn when a tab
  asks for them (`display_layer`, under the edit lock: they took 45-65 ms of every edit), and the edit
  leaves the arrays it wrote in the layer cache (`_install_layers`: loading them again took 20-30 ms of
  the next commit). The part's shine cues are measured on a window (the whole-image pass took
  1-2.5 s) against the residual's glint reference, cached per job (`_spec_q`), the protect mask reuses
  the photo's cached hue and chroma (`refine._photo_hue_chroma`), the work image is the prompt
  session's, and the studio's empty prompt when the tool opens warms these, the layers and the
  modules the first commit loads (`_warm_commit`).
- **Edits.** A merge or move with a target (`into`, or the move's group) is what the user asked for
  whatever the sizes: a colour region moved or merged into a user part joins it, a user part merged
  into a detected part becomes an instance of it (the far caliper into "Brake caliper" makes "Brake
  calipers"), into a colour group it is dissolved; the group merged into keeps its own paint, or none
  (carried by membership, a painted part merged into an unpainted colour painted the whole colour).
  Without a target the part follows the Part badge (`normalize`: area). Every edit keeps every other
  group's lock and background flags as they are (only the automatic rules of the analysis and a
  regroup set them, and a regroup gives back the user's own). Remove part is `merge {group_ids:
  [part], dissolve: true}` and undoes the carve: every pixel of the region the carve made goes back to
  the region it came from (`_undo_carve`; those of the region that gave the most stay in the carve's
  region, which becomes that region's twin in its group, with its tag, so no region id is emptied or
  renumbered, and the seed's `origin` of the twin is its donor's); every group the part took in whole
  is a group of its own again, with its name, lock and background flags and its paint of then (a
  colour group too, and each instance of a split part: they came back as one group with one paint);
  a region of a group it took only in part goes back to the group now holding its `homes` region; a
  region wholly inside the part takes back the tag it had. Removing a part cut mostly out of another
  user part gave the other one back all of its pixels (it dissolved it: the whole cut, mostly that
  part's pixels, joined it as a colour region). The part's paint goes with it (carried by membership
  it painted the whole group it went back to), no other group changes its name or paint; the seed's
  tags and the registry follow (`sync_seed_tags`), so a regroup does not bring the part back. Remove
  part 0.11-0.13 s.
- **Find part** (`find`): first the part groups the phrase names (`_names_part`: every word of the
  phrase, plurals and spellings folded, is a word of the group's name, its kind's label or plural,
  its kind, the head noun of one of the vocabulary's prompts of it, or a `FIND_SYNONYMS` word of one of
  those, or the phrase is two words or more of one prompt: "caliper" names "Brake caliper" and a
  drawn "Far caliper", "muffler" "Exhaust", "saddle" "Seat", "shock absorber" "Shock spring", and
  "motorcycle" alone no "motorcycle seat"; at most three, `existing`, taking one selects the group),
  then OWLv2 on the image and its tiles for the phrase and up to three more phrases (`_find_phrases`:
  `FIND_SYNONYMS` and the vocabulary's prompts of the kind it names; on the Ducati "mirror" found
  nothing, "rear view mirror" the mirror first; Florence-2's phrase grounding without OWLv2), the boxes
  ranked and de-duplicated, each prompted as a SAM box and refined like a Select part box, five
  candidates in all, of distinct masks (a detector mask that is an existing group's is dropped),
  ranked by detector score to the 0.5 x SAM score squared (halved mostly on the background), each with
  the part group it overlaps (`matches`, IoU 0.5, `named`: whether the phrase names it). A candidate
  that is a part group of a kind the phrase does not name comes after the others (Find "spring": the
  detector's exhaust can, the Exhausts group, came before the spring; now it is not among the five).
  The groups and the group map are read in one edit generation (`_snapshot`: an edit landing between
  two reads pointed `matches` at the wrong group). 0.7-1.1 s (OWLv2 0.26-0.57 s; 2.5-2.8 s when OWLv2
  loads first). OWLv2 scores painted calipers at 0.05-0.12 (see the wheel second look), so the
  ranking leans on SAM's score: a detected caliper is offered by its name first.

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
                 group_map: np.ndarray, groups: list[ColorGroup], device=None,
                 islands: np.ndarray | None = None, protect: np.ndarray | None = None,
                 reference_long_side: int | None = None, glints: np.ndarray | None = None,
                 neutral: np.ndarray | None = None): ...
    def render(self, mapping: Mapping, options: RenderOptions) -> np.ndarray   # uint8 sRGB
    def render_at(self, long_side: int, mapping, options) -> np.ndarray         # resized layers, cached per size
    def recolor_albedo(self, mapping, options, long_side=None) -> np.ndarray    # float32 linear
    def white_glints(self) -> np.ndarray      # float32 HxW: the white paint's glints (rule 7e), for an export
    def neutral_weights(self) -> np.ndarray   # float32 [G, 2]: each group's neutral / white-paint weight, for an export
    def neutral_sources(self, mapping, options=None) -> tuple[int, ...]   # mapped groups a neutral rule touches
    def update_groups(self, groups, protect=None) -> None   # new flags / protect mask, same groups: no rebuild
    def free(self) -> None
def render_once(albedo_lin, shading_lin, residual, group_map, groups, mapping, options,
                islands=None, protect=None, reference_long_side=None, glints=None, neutral=None) -> np.ndarray
def recompose(albedo_lin, shading_lin, residual) -> np.ndarray                  # the identity reference
```

Algorithm (torch, GPU). Pixels and group colours live in OKLab, from the 2.2-gamma linear
RGB (CIELAB's hue lines bend toward purple on desaturated blues: the navy repaint of the
yellow BMW read violet); CIELAB thresholds are translated at OK chroma ~ CIELAB / 350. Each
rule below was a visible defect before it existed; the module docstring gives the details.
1. Coverage `m`: the mapped indicator snapped to the photo with a colour guided filter,
   maxed with the hard label, ramped outward only (`feather_px`), maxed with the hard
   label again (the ramp left 1-3 px slivers of a label half painted). The ramp is gated
   by how much of the old paint each pixel's albedo holds, its projection on the line from
   its own group's colour to the paint's (none below 0.03, all from 0.15; kept where the
   two colours are too close to tell): ungated it painted the first 1-3 px of every
   neighbour, a grey contour around the BMW's white tail lens and a haze along every
   silhouette against a white backdrop. Where the photograph's own edge is soft, the ramp is
   let through anyway (edge width = the local sRGB range over a 7x7 window divided by the
   largest 1-px step, open from 2 to 3 px at the reference resolution): the decomposition
   sharpens the albedo at a defocus blur, and the gated repaint of the Exia's defocused feet
   read as a hard cut-out. Studio shots measure 1.5 px at their repaint boundaries, the
   Exia 2.6. Decal islands are never entered unless their own group is mapped.
2. Albedo shift per mapped group (target T, source A = the group's `albedo_lab` in OKLab, or
   its `ref_lab` for an instance of a part split by instance: every instance is painted from
   the part's colour):
   `ab' = T_ab + s·R(θ)·(ab − A_ab)`, `s = min(1, C_T/C_A)`, `R(θ)` the source-to-target hue
   turn (faded out near neutral). Lightness on a CIELAB-like toe of OK L (pure OK L crushed
   black repaints): `L' = T_L + slope·(L − A_L)`, slope `(T − black)/(A − black)` below the
   anchor and `min(1, (T − black)/0.2)` above it (texture turned into grey blotches on
   black). At sensor-clipped white highlights only the radial part of `ab − A_ab` is kept
   (the albedo is magenta-shifted there; rotated into navy it went cyan). `flat` mode sets
   `T`, `texture` blends. Out-of-gamut colours lose chroma at fixed OK L and hue.
3. `shading' = pivot · (shading / pivot) ** shading_strength`.
4. Bounce light: per pixel, the light's tint beyond the scene illuminant (white balance on
   well-lit neutral surfaces) is split along the old paint's chroma; the aligned part is
   turned and scaled like the albedo (a navy tank's shadows went teal); at clipped
   highlights the light is the illuminant.
5. Residual: the positive residual is split at its achromatic floor; the coloured excess
   is rebuilt as a multiple of the repainted product (red painted black came out maroon),
   and on the repainted labels a darker target keeps only `follow ** 0.5` of its energy
   (`follow` = new / old product luminance; kept whole, the yellow BMW's residual lifted a
   #1b2a57 navy to a medium royal blue, four times the target's lightness); the floor is
   kept where it is a glint and faded where it is a veil, and within 2 px of a
   group boundary it may not exceed the interior's (it drew a hairline along every
   silhouette of a dark repaint). The negative residual scales with the new product and is
   dropped at clipped highlights. `residual_tint` lerps toward the target.
6. Paint envelope on the composed repainted pixels: a black floor (0.24 % reflectance lit
   by `S·(S/S_med)^0.6`, so black keeps its form); a gloss floor (never below the photo's
   white specular, split dichromatically against how white the paint looks at that
   brightness, band-limited, min-blurred, soft-thresholded; how white a paint looks is
   measured over its group's pixels, and the instances of a part split by instance share the
   part's measurement, `Renderer._pools`: measured on its own, the BMW's rear caliper, out of
   the light, changed 157 px of the render when its part was split; the boundary band of the
   residual floor and the gloss floor reads the instances as one part too, since their seam is
   a group boundary only after the split: the robot's two feet touch through their rim, and the
   seam changed 43 px by up to 13 levels); OK hue held within 4 deg of the
   target; chroma capped at `1.25·C_T + 0.003`; shadows of a light target keep
   `0.9·C_T·L/L_T`. In the outward ramp the envelope is weighted by rule 7's permission.
7. Reflections: pixels outside the repainted groups that still show the old paint get the
   paint's OK hue turn and chroma scale at fixed OK L, weighted by OK chroma (half at
   CIELAB 12), hue (30 deg window, stopping 20 deg short of unlocked coloured groups;
   60 deg within 5 px of the paint), distance (full within 70 px, none beyond 140),
   `1 − m`, a lightness ceiling (the paint's p90; islands exempt), the `protect` mask,
   24 px around locked groups, and a permission (never repainted, locked or coloured
   unmapped groups, nor an unmapped group of CIELAB L >= 30 at least half of whose photo
   pixels pass this colour test: an object of that colour, like the Exia's warm backdrop
   behind its repainted gold parts, which went grey and blue). Island pixels rebuild the
   old paint's excess in the target colour; the result is held within 8 deg of the target
   hue and blended through neutral. Sources: groups repainted from one colour to one target
   are one source (the instances of a split part, whose reference is the part's), with the
   union of their pixels. A small source (below `REFL_SMALL_FRAC`, 1 % of the image) reaches
   only `REFL_REACH_K` (1) x the square root of its area (a windowed distance transform per
   source): painted alone, the Ducati's 3.5k px shock spring tinted the reflections of its
   gold frame 86 px away (2 073 px changed beyond 8 px of the spring, now 3-6 over four fresh
   analyses). And an unmapped,
   unlocked coloured group of the source's own hue (within `REFL_HUE_MARGIN`) at least
   `REFL_SAME_HUE_AREA` (half) the source's size closes the source's hue window on both sides:
   every pixel of that hue may be that object's reflection (the gold frame beside the gold
   spring); a crumb of the hue only stops the window short of its own hue, as before. The
   own-colour permission above reads the window as cut, before a closure. Painting a main
   paint (1 % of the image or more, no same-hue object half its size) renders exactly as
   before: 0 px changed on the six end-to-end photos.
8. `out = linear_to_srgb(clip(result, 0, 1))` → uint8; with an empty mapping this is the
   recomposition bit for bit.

Neutral sources (white, off-white, a light grey; `NEUTRAL_*`, `GLINT_*`, `ISLAND_*`, `EXPOSURE_*`, the
module docstring has the numbers): on white paint the Careaga layers put the white's brightness mostly into
the shading, give the albedo Y 0.6-0.8, and leave a broad achromatic residual (2-18 % of the photo, up
to 50 % on a sun-lit one) on every lit face, which rules 2, 5 and 6, written for saturated paints, kept
as glints: 26-63 % of a navy repaint's lit luminance stayed neutral grey (cornflower lit faces, black as
grey marble, a pastel blown to white). That leftover is the white paint's own diffuse light. A source's
neutral weight is 1 up to CIELAB chroma `NEUTRAL_C0` (10) and 0 from `NEUTRAL_C1` (18), times the
larger of an albedo-lightness ramp (`NEUTRAL_L0/L1` 45/60) and a photo-white-share ramp
(`NEUTRAL_WHITE_S0/S1` 0.04/0.10), times a white-paint gate, 0 for a group tagged chrome. The gate: a
grey that glints (`NEUTRAL_GLOSS_G0/G1` 0.12/0.25 of its pixels sensor-clipped or holding a neutral
residual above half the image's 99th percentile, the analysis stage's `Region.glint` cue measured again
on the layers, so a job analysed before that cue gets it too, and counted softly around both cuts,
`GLINT_PX_*`: the largest channel from 0.97 to 0.99 sRGB, the residual from 0.3 to 0.7 of that
percentile; with hard cuts a streak whose residual sat at one value moved a glossy grey's weight from 1
to 0, and its render by 81 levels, for a 0.005 step of the streak) is metal or a gloss, whose floor is its
reflections, and keeps the saturated-paint rules unless its albedo (`NEUTRAL_PAINT_L0/L1` 72/82) or its
photo's white share (`NEUTRAL_PAINT_S0/S1` 0.10/0.45) says white paint: weight x (1 - (1 - white) x gloss).
Under the white-paint rules the BMW's cast fork leg and disc (L 64, 30 % glints) read as matte plastic on
red, a bicycle's glossy white-and-silver parts (L 74, 20 % white, 27 % glints) lost their sheen, a concrete
wall (27 % glints by that measure) its texture, and each became a flat blob on black; the RX-78's white
armour glints on 27-35 % of its pixels and stays white paint (L 85, 56 % white). The weight is blended per
pixel like the other group parameters and weights every rule below, so a saturated source (and black,
and a dark or glossy grey, weight 0) renders bit for bit as if the rules did not exist. Per mapping it is
also faded by the target's CIELAB distance to the source colour (`NEUTRAL_NEAR_DE0/DE1` 2/12): a map to a
group's own swatch colour renders exactly as under the saturated-paint rules, near the photo (the
exposure bound had taken 43 L* off the lit faces of the Unicorn's grey-albedo armour for such a map).
Every ramp is a smoothstep, so the render is continuous in the source's colour, white share and gloss and
in the target. A full-resolution export takes the preview's weights (`Renderer.neutral_weights`) with its
glints.
- Rule 1: the decal-island pixels that hold the paint in their albedo (their share of it against the
  decal's own colour, which must be clearly another), connected to the repainted label (a piece
  touching it within 1.5 px, reaching at most 24 px) and no lighter in the photo than the paint along
  their contact, are repainted with it as far as the mapping's neutral weight goes
  (`Renderer._island_entry`, cached per mapped set). The analysis took the white leather between the
  sneakers' FILA letters into the letters' decal, which stayed pale speckle on navy; a white reflection
  in the Alpine's mirror glass next to the body is lighter than the paint and is left alone.
  Connectivity, not a distance, decides: the export's edge-snapped labels closed the paint's specks
  inside the decal and left leather 13 px from the paint.
- Rule 2: above the anchor the slope is `NEUTRAL_UP_K` (0) of the room ratio (nothing on a white paint
  is lighter than the paint).
- Rule 5: the neutral floor is rebuilt in the new paint like the product (`follow **
  FLOOR_DIFFUSE_GAMMA`, 1), and the coloured excess follows the same way (`NEUTRAL_EXCESS_GAMMA` 1).
- Rule 6: before the black floor, the exposure bound: where the photo is white no repaint of albedo A'
  reflects more than photo x A' / `WHITE_REF_Y` (0.8), the light read from the brightest white pixel of
  the group within 4 px. It acts as far as the group's photo shows white paint, and the lighter the
  albedo the more white it takes (a share ramp of 0.04-0.10 for an albedo of Y 0.15 or less, 0.20-0.40
  from Y 0.45: a dark albedo under a white photo contradicts it, a light grey may be that grey), and at a
  pixel as far as its neighbourhood is white (`EXPOSURE_NEAR`, a quarter of it: one white pixel set it on
  over a 4 px disc and dotted the Unicorn's grey faces with dark discs). A small clipped spot on paint that
  is not white is no source of that light (`Renderer._small_clips`, as far as the whiteness of the paint
  around it rests on 2-8 px, `GLINT_STAT_PX*`; its per-spot sums are float64 on the CPU, so two builds of a
  job give the same field: the GPU's weighted bincount differed by 4e-7): the Unicorn's sun streak switched
  the bound on around itself and sat in a dark pad at 0.56 of the face; now 1.12 on navy (the photo's
  1.10). A neutral group's own saturated-paint white-specular estimate is scaled by 1 minus its weight.
  And (e) the white paint's own glints come back on its labels as white light (`Renderer._white_glints`):
  a sensor-clipped white spot, weighted by smoothsteps of its area (4-8 up, 120-180 px down at the
  reference resolution), of how far it stands out from the lit paint of its own group 2-6 px around it
  (1.20-1.50 up, 3.5-4.5 down, the ring's albedo that paint's), of how little of that the shading layer
  explains (its standout over the shading's step, 1.28-1.44: a lit bevel, crease or rim is in the
  shading, 0.7-1.2x on the sneakers' creases and the RX-78's vent rim, the glints 1.44-2.2x) and of how
  much brighter it is than the brightest unclipped paint of that ring (its 90th percentile, 1.05-1.20:
  the RX-78's one kept "glint" was a clipped corner of a lit face, 1.10x it), the last as far as the
  shading steps up under the spot (core over ring, 1.08-1.20: on evenly lit paint the bright ring pixels
  are the glint's own halo, and the model ship's brightest lamp dot, 1.04x its halo, was painted) and as
  far as that percentile rests on 2-8 px. Every ramp ends where the hard tests that first kept the glints
  cut, or just past the weakest glint they kept (8 px, 1.5x, 1.44x), so those glints are kept whole:
  centred on the cuts, the ramps had halved the glints just past them (the Alpine's roof streak, 1.54x,
  read as a pale blue stripe on navy). It is added as the photo's brightness on the whole clipped core
  and the photo's own profile over the ring around it, squared, faded out 4 px away, times that weight:
  a glint dims below the cuts instead of switching off (a hard cut flipped a 45 px glint by 237 levels in
  one step of 0.25 L*). Found once at the layers' resolution and resized for the preview; a
  full-resolution export is handed the working-resolution field (`glints`).
- Rule 7: a neutral source reflects with weight 1 minus its weight (its colour cast is the light's), and
  every source with its hue confidence (CIELAB chroma 2 to 8): a near-neutral paint's reflections come in
  with its hue instead of switching on at full weight at chroma 2, which with a partly neutral glossy grey
  moved the pixels of a decal next to it by 25 levels in one step of 0.1 chroma. A saturated source is
  unchanged.

Measured on six white photos x six targets (scratch/whiteexp/exp, re-run on the product code in
scratch/whiteint/r3): washout 9.48 -> 0.45, black lift 12.89 -> 0.47, specks 388 k -> 18.8 k px (14.9 k of
them outside the kept glints; a kept glint on a dark repaint is a white peak the photo's own peak does not
match, and the speck count includes it), bleed 238 k -> 6.6 k px (1.2 k of them the decal pixels rule 1
repaints), white rim 65.8 k -> 43.1 k px (40.1 k before the near-source fade: 2.6 k of the difference is the
Unicorn's shaded armour, whose albedo is the grey target's own colour, keeping its photo for that target).
Glints: every glint the first hard tests kept is kept whole (the Alpine's roof streak, the model ship's lamp
dots, its brightest one included, the Unicorn's sun streak, the Nu's socket rings): of the clipped spots
that stand 1.5x above their ring, the share still near-white is 0.46 on the Alpine and 0.93 on the model
ship for every target (0.00 and 0.35-0.42 with the ramps centred on the cuts, 0.46 and 0.82-0.93 under the
hard tests); spots just below the cuts are kept in part (the ship's dimmer lamp dots at about half, lit
edges at a fifth or less), and the RX-78's lit-face corner at a quarter (0.24: the area ramp no longer
halves an 8 px spot). Real grey jobs (read-only): the BMW's cast metal on red keeps its top 10 % at
OK L 0.80 (frozen 0.80, before the gate 0.49), the bicycle's glossy parts 0.65 (0.67, 0.50), the concrete
wall 0.69 (0.70, 0.51). The saturated controls, the six fixture jobs' halo harness and 196
saturated-source renders are unchanged bit for bit; steady renders unchanged, the first preview render of
a renderer 35-55 ms slower than before the white-paint work (the small-clip and glint searches), the first
working-size render after it up to 25 ms. Limits: broad, unclipped reflections of a white
paint (the sky on the Alpine's flank) are repainted with it, so a glossy white car repainted dark reads
satin apart from its sharp glints; black on white paint is matte; a 1-5 px white rim stays where a label
stops short of the edge (segmentation); a matte-reading silver (few glints, the Ducati fixture's lit
floor strip and fork tubes) still gets the white-paint rules; the exposure bound acts where the photo is
white, so a white face can repaint darker than a grey-albedo face of the same paint beside it.
`tests/test_engine_neutral.py` covers the neutral weight and its gate (a glossy grey keeps its
reflections, glossy white paint stays neutral), a map to the swatch colour (the photo), each rule (lit
faces reach the target, a small clipped glint stays white and one just past the old cuts is kept whole,
its dimmer clipped pixels included, a glint with a bright halo on evenly lit paint is kept, while a lit
bevel, a clipped face, white lettering and a lit face's corner are repainted, no pad around a glint, no
disc around a lone white pixel, the export keeps the preview's glints and weights, the decal's paint is
repainted and a lighter pixel is not, a near-neutral source's reflections come in with its hue),
continuity (sweeps of the source's lightness through the glint's ramp and of its chroma, of the target's
distance from the swatch colour, of the sun across the white-share ramp, of a uniform streak across the
gloss gate, a glint's standout) and that saturated sources are unchanged. `tests/test_export_reference.py`
checks the export's reference with the real engine: the preview's renderer is reused only for its own
snapshot, no glints without a neutral source, `(None, None)` on CUDA out of memory.

Pixel distances are defined at `reference_long_side` (the layers' own by default) and scale
with the render size. Per-mapping tables (distance transforms, quantiles) are cached in a
small LRU per renderer.

Performance target: preview (1024 long side) render < 60 ms after the first call
(measured 6-8 ms on the two motorcycles with the GPU otherwise idle); full 4K render < 1 s.
The first render of a renderer builds the per-image tables (0.1-0.4 s once); a lock or
background toggle keeps the renderer (`update_groups`), a merge, split, move or regroup
builds a new one. `tests/test_engine.py`:
identity reproduces the recomposed original bit for bit (also with islands and protect),
a flat repaint hits the target (ΔE < 3), every option combination renders finite, and one
scene per rule above (reflections never touch locked / protected / coloured groups,
islands are never entered, the black floor keeps form, the white estimate and highlight
mask, the boundary hairline, the pastel shadow floor, a small source's reach, the same-hue
closure and a crumb that does not close, the instances of a split part rendering like the
part, sharing its gloss statistics and, where they touch, having no seam in the boundary band).

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
def segment(job, body) -> dict           # POST /segment: a SAM 2 prompt (User parts, §3.2)
def add_user_part(job, body) -> dict     # POST /groups/from_mask: the prompt run again and carved
def find_parts(job, body) -> dict        # POST /find: OWLv2 boxes prompted as SAM boxes
```

Group edits of one job are serialised (`Job.edit_lock`, held for the whole edit, separate
from `gpu_lock`): two lock toggles 20 ms apart used to keep only the second, and a toggle
racing a merge wrote the pre-merge groups back. Every edit writes the grouping inside a
section that moves a per-job generation (a sequence lock), and a renderer build or a
full-resolution export reads the layers and the groups of one generation (rebuilt when an
edit wrote meanwhile), so neither mixes the group map of one grouping with the groups of
another, nor caches such a mix.

`analyze` order: ingest (save original, make work + preview images) → intrinsic →
segment (SAM) → regions (lettering and named parts, `build_regions`, part recovery, the
matte cut, the detected parts, the subject check, the backdrop decisions) → groups (clustering, then
`refine.refine_groups` with `matting.snap_labels` and the junk pruning); each stage sets
progress and a human message
("Finding parts with SAM 2 · 64 points/side · crop layer 1/2"), catches exceptions into
`status: "error"` with the message, and records seconds. A single worker thread
processes jobs FIFO (the GPU is shared). A queued or analysing job records its server
process in `owner.json`; at start-up `resume_pending` re-queues only the jobs whose owner
is gone, so a second server on the same data directory never takes over a live analysis.
A server started from code older than owner files claims nothing, but its analysis keeps
rewriting job.json: while an unclaimed queued or analysing job was written within the last
90 s, every unclaimed job is left to that server and looked at again every 30 s (a job it
finished is taken from disk; the rest are resumed once they have all been quiet for 90 s).
SAM, Intrinsic, ViTMatte, Florence-2, BiRefNet and OWLv2 are lazy singletons: by default they load
on the first job that needs them and a background watchdog drops them again after
`config.IDLE_UNLOAD_S` (120 s) with none running, so the GPU only holds their memory while
the app is in active use; `serve.py --warmup` preloads them at startup instead, but they
are still unloaded on the same idle timer afterwards.

Job artifacts under `data/jobs/<id>/`:

```
job.json          # meta (see API)          original.jpg   # full res (png if upload was png)
work.png          # working res             preview.jpg    # preview res
albedo.npy shading.npy residual.npy         # float16, working res, linear (the segmentation
                                            # stages use this float16-rounded albedo too)
labels.npy group_map.npy                    # int32, working res
islands.npy protect.npy                     # bool, working res: engine masks (absent on older jobs)
regroup.npz                                 # the regions stage's label map + each region's origin
                                            # + its 'part' regions + each region's backdrop decision
                                            # + each input region's source and detected-part tag
                                            # (part_kind / part_label / part_plural / part_instance)
                                            # + the junk pruning's parameters, the object mask and
                                            # the parts' SAM masks (packed bits)
                                            # (refine.regroup_refined); absent on older jobs;
                                            # a user part is carved into it too (its own tagged
                                            # input region, the new region descending from it)
user_flags.json                             # the user's lock / background choices per region and
                                            # the user parts (`parts`: kind -> label, regions,
                                            # donor regions, the tags of the regions it covered,
                                            # the groups it took in whole (regions, name, flags,
                                            # paint), `homes`, `new_id`; `inside` for one another
                                            # user part took in whole)
parts/<kind>.npz                            # a user part's carve: over the box of the region it
                                            # made, the region each pixel came from (Remove part)
fullres_albedo_<method>.npy                 # float16, working res: the first full export's
                                            # full-res albedo, area-downsampled (drift check)
owner.json                                  # while queued / analysing: the owning server process
layers/albedo.jpg layers/shading.jpg layers/residual.jpg layers/regions.png
layers/groups.png layers/edges.png          # display layers, working res
thumbs/preview_w512.jpg                     # cached thumbnails (GET layers/<layer>?w=), made on demand
ids/regions.png   # RGB-encoded region ids: id = R + 256·G + 65536·B
ids/groups.png    # R = group id (0..255), G = B = 0
exports/<name>.png|jpg
```

`regions.json` carries every region's `backdrop` decision, `shiny` and `glint` shares,
`chrome` advisory and detected-part tag (`part_kind`, `part_label`, `part_plural`,
`part_instance`; empty / -1 for every other region; `user_<n>` for a user part, whose new region
has source `'user'`); `job.json` carries `ignore_background`,
`panel_rule` (the panel rule the groups' view was made with, `grouping.PANEL_RULE`; a job of an
older rule, or none, is annotated again in memory when served) and, per group, `shiny`,
`glint`, `finish`, the part-group fields (`part`, `part_label`,
`part_plural`, `part_instances`), the panel view (`minor`, `parent`) and, for an instance of
a part split by instance, `ref_lab` (the part's albedo, null otherwise). A record written
before any of these fields existed loads with their defaults (no part, not minor), so an older
job groups, regroups and renders exactly as before. A stored `finish` badge is checked against the glint rule again whenever the
record is read (`ColorGroup.from_dict`, which `Job.load` runs over every group), so a job
analysed when the badge followed the broader highlight share loses its stale badges on load
rather than at its next regroup.

### 3.7 HTTP API — `recolor/server/app.py`, `serve.py`  (owner: SRV)

FastAPI. Static `web/` mounted at `/` (index.html at `/`, no caching in dev). JSON
errors as `{"error": "...", "detail": "..."}` with proper status codes. CORS open.

| method | path | body / query | returns |
|---|---|---|---|
| GET | `/api/health` | | `{ok, device, gpu, vram_total_mb, vram_used_mb, models:{sam2, intrinsic, vitmatte, florence, birefnet, owlv2} ('cold'|'loading'|'ready'; the last four also 'unavailable'), jobs, version}` |
| GET | `/api/samples` | | `[{name, url:"/api/samples/<name>", thumb:"/api/samples/<name>?w=320", width, height, title, license}]` |
| GET | `/api/samples/{name}` | `?w=` optional | image |
| GET | `/api/jobs` | | `[JobSummary]` newest first: `{id, name, created, status, thumb, width, height, n_groups}`; `thumb` is `/api/jobs/<id>/layers/preview?w=512` |
| POST | `/api/jobs` | multipart `file` **or** JSON `{sample: name}`; optional fields `detail`, `intrinsic`, `max_groups`, `delta_e` | `Job` (status queued) |
| GET | `/api/jobs/{id}` | | `Job` |
| DELETE | `/api/jobs/{id}` | | `{ok}` |
| GET | `/api/jobs/{id}/events` | SSE | events `{type:"stage", stage, state, progress, message}`, `{type:"status", status}`, `{type:"groups", groups}`, `{type:"done"}`, `{type:"error", message}`; replays the current state on connect; heartbeat comment every 15 s |
| GET | `/api/jobs/{id}/layers/{layer}` | layer ∈ `original, work, preview, albedo, shading, residual, regions, groups, edges`; `?w=` optional | image, or with `w` a cached JPEG thumbnail at that width (never upscaled); `regions`, `groups` and `edges` are drawn on the first request after an edit (`pipeline.display_layer`) |
| GET | `/api/jobs/{id}/ids/{kind}` | kind ∈ `regions, groups` | PNG (see §3.6) |
| POST | `/api/jobs/{id}/groups/merge` | `{group_ids:[..], into?, dissolve?}`: `into` (one of the ids) keeps its name and its own paint (or none) and takes the others whatever their sizes (a user part merged into it joins it, into a detected part it becomes an instance of it); `dissolve` removes the user parts merged; `{group_ids:[part], dissolve:true}` alone is Remove part (its paint goes with it; see User parts, §3.2) | `Job`; Remove part adds `removed_part:{name, restored:[names of the groups given back], home}` |
| POST | `/api/jobs/{id}/groups/split` | `{group_id, k, mode?}`: `mode` `"colour"` (default, k-means on the albedo into `k`) or `"instances"` (a part group with several instances into one group per instance, `grouping.split_instances`; 400 on any other group) | `Job` |
| POST | `/api/jobs/{id}/groups/move` | `{region_ids:[..], group_id}` | `Job` |
| PATCH | `/api/jobs/{id}/groups/{gid}` | `{name?: str, locked?: bool, is_background?: bool}`: the name trimmed, 1 to 48 characters; 400 on any other type or length, as `PUT /state` does for `ignore_background`; a `null` leaves that field as it is | `Job` |
| POST | `/api/jobs/{id}/regroup` | `{max_groups?, delta_e?}`: `max_groups` an integer 1-256 or null (auto), `delta_e` 0.5-100, as for a new job (400 otherwise; a stored option out of range falls back to the default) | `Job` |
| POST | `/api/jobs/{id}/segment` | `{points:[[x,y,1\|0]..], box:[x0,y0,x1,y1]\|null, multimask, pick?, crop?}` (work pixels); `{}` prepares the embedding only | `{mask:{png, bbox, area, area_frac, score, index, refined}, alternatives:[..], pick, crop, steps, size, timings, embed:'computed'\|'cached', ms}`; 400 bad prompt, 409 not ready, 503 + `Retry-After` while the GPU is busy |
| POST | `/api/jobs/{id}/groups/from_mask` | the `segment` payload + `name?` + `take?` (up to 8 ids of groups under the selection that the part takes in whole) (run again here) | `Job` + `created_group`, `created_part:{kind, name, area, regions, refined, took_in:[names of the groups taken in whole], replaced:name\|null, ms}` |
| POST | `/api/jobs/{id}/find` | `{text}` (1-60 characters) | `{text, phrases, detector:'owlv2'\|'florence'\|null, candidates:[{box, score, rank, phrase, mask, matches:{group_id, name, iou, named}\|null, prompt\|null, existing, group_id?}], detect_ms, ms}` (up to 5; `existing` ones first, `prompt` null; one that is a part group the phrase does not name, `named` false, last) |
| POST | `/api/palettes` | `{prompt, n_colors}` | `Palette` |
| GET | `/api/palettes/{pid}` | | `Palette` |
| GET | `/api/palettes/{pid}/sources/{i}.jpg` | | image |
| POST | `/api/jobs/{id}/mapping/suggest` | `{colors:[hex..], strategy, keep_background?}` | `{mapping:{gid:hex|null}}`; background groups are left out while the job ignores its background |
| POST | `/api/jobs/{id}/render` | `{mapping, options}` | `image/jpeg` preview, header `X-Render-Ms` |
| POST | `/api/jobs/{id}/export` | `{mapping, options, quality:"work"|"full", format:"png"|"jpg"}` | `{url, width, height, ms}`; 503 with `Retry-After` when the shared GPU has no room (after one retry) |
| GET | `/api/jobs/{id}/exports/{file}` | | file, `Content-Disposition: attachment` |
| PUT | `/api/jobs/{id}/state` | `{mapping?, render_options?, palette_id?, ignore_background?}` | `Job` (persists UI state) |

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
 "palette_id":null, "mapping":{}, "render_options":{}, "ignore_background":true}
```

`serve.py`: `uvicorn` on `0.0.0.0:config.SERVER_PORT`, prints the LAN and Tailscale
URLs (detect like SaxScope's `serve.py`: UDP-connect trick for LAN, `tailscale ip -4`),
starts the idle-unload watchdog (models load lazily unless `--warmup` is passed), sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch is imported unless the
environment already says otherwise (see §1), and
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
   and a "Try a sample" strip from `/api/samples` whose heading carries the same detail
   selector (the drop zone's is hidden until a file is picked, and the samples use the
   level too). Recent jobs row from `/api/jobs`.
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
     - *Groups*: an "Ignore background" switch (on by default for a product shot, off
       when the analysis found a scene with a fragmented background; background groups are
       dimmed and read as locked while it is on), then the list of swatches (albedo
       color, name, area bar, region count), lock toggle, background toggle and a
       Background label (a label only: the toggle is the one control, since a badge that was
       itself a button sat at the row's centre and a click meant to select the row unmarked
       it), a finish badge (shiny / chrome, advisory; the title line wraps, so a long name
       keeps its width and the badges take a second line), rename inline, drag one group onto
       another to merge (a toast says when the merge made a painted group part of the ignored
       background), "Split" and "Auto-regroup" (slider for the colour-group count; the hint
       says that a detected part, a locked part or a decal keeps its own group) actions. The
       list keeps its scroll position across a lock or background change, and the footer's
       buttons wrap in a narrow inspector. The rows come in four sections, largest first in
       each (`sectionOf` in `panels/groups.js`; a job with nothing but colour groups shows no
       headers): **Parts** (part groups: the kind's name, a green Part badge with a tooltip,
       icon-only in a narrow inspector, "N instances" in the meta line; Split on a part with
       several instances splits it by instance, `mode: "instances"`, which its tooltip says
       while the label stays "Split": a wider label pushed Auto-regroup onto a second line
       whenever such a part was selected; Merge joins the instances again under the kind's
       name), **Colours**, **Minor** (tiny groups in the colour of the group next to them,
       mostly its shadow or reflection; collapsed under a divider with its count, expanded by a
       click; each row says "next to <parent>" and carries a one-click "Merge into <parent>"
       button; never hidden from painting) and **Background** (last; collapsed while the
       background is ignored, until the user opens or closes it by hand). A group selected on
       the canvas opens its section. The selection pill says "part, N instances" or "minor",
       and for several groups "N parts selected", "N colours selected" or, mixed, "N groups
       selected". A colour split that finds one colour (a one-region part) says so in a toast
       ("Nothing to split: Shock spring is one colour") instead of reporting a split. A share
       too small to show at two decimals reads "<0.01%", never "0.00%". Above the list, *Find
       part*: a phrase ("spring") asks `find`; the candidates are outlined on the canvas in amber
       with their numbers and listed as chips (size, OWLv2 confidence, the part group a candidate
       is or overlaps), a hover or a keyboard focus on either highlights both (the chips are built
       once per answer: rebuilt on every hover, a focused chip was replaced under the focus), a
       click on either takes the candidate into Select part (its prompt, named after the phrase),
       or selects the group when the candidate is a part group the phrase names. A drawn part's row has a
       Part badge with the wand icon and a Remove button (the part goes back where its pixels came
       from); a row dropped onto another merges into it (`into`).
     - *Select part* (the toolbar's wand button, `S`, or an Alt-click on the image): a click adds a
       positive point, a Shift-click or a right-click a negative one, a drag draws a box; wheel and
       pinch zoom, a right or middle drag pans. Every change asks `segment` (debounced 24 ms, the
       in-flight request aborted, one retry after a 503's `Retry-After`) and the viewer outlines the
       mask in green with the points and the box; the selection pill becomes the tool's pill (what
       Enter will do first, then the part's share of the image and the prompt; a name field, Take all,
       Make group, the other shapes SAM offers for one click, Undo, Cancel). Enter commits, Backspace
       undoes a point, N cycles the shapes (Tab cycling them trapped the keyboard focus in the viewer),
       Esc cancels; a focused button, tab, switch or slider keeps Enter and Space for itself, so
       pressing the pill's Undo or a Find chip never commits or peeks. A double click is two points
       there, not a zoom. The keys are in the status line's hint, not on the pill's line (at 1024 px
       the line cut them off; a line the pill cuts with an ellipsis is there in full on hover), and on a
       touch-only screen the hint names the pill's buttons instead. The pill says what Enter will do
       to the groups the mask covers (the group id map counts the mask's pixels): "replaces Far
       caliper", "takes in Bolt" (a part group covered 90 %, a colour group covered whole), or "cuts
       Shock spring (32 % stays)" for a part group it covers half or more of but not 90 %, when
       **Take all** (plus icon; an icon button on a phone) makes all of that group join: the outline
       grows over it (`viewer.spriteWithGroups`), the line says "replaces Shock spring", and the
       commit sends `take`. Enter pressed while the outline of the last click is still on its way
       (or retried while an analysis holds the GPU) waits for it and commits it once it shows
       ("Making the group as soon as the outline is drawn"), so a commit never makes a mask the user
       did not see; clicks, boxes, Backspace and N while a commit runs are not taken ("Making the
       group · one moment"). The tool prepares the embedding when it opens. Enter posts
       `groups/from_mask` with the answer's `pick` and `crop`, reloads the id maps, and selects the
       new group, ready to paint. The status line shows the last round trip ("SAM 2 25 ms"); its hint
       shrinks with an ellipsis and the timing chips never wrap. On a phone the tool's pill takes two
       rows (name and facts above, the name field and the buttons below: in one row the field shrank
       to four letters). A Find answer that arrives after the tool closed, a newer Find or a click on
       the image is dropped (a token and an abort), and so is a mask answer after Backspace emptied
       the prompt. The candidates' numbers sit on the mask (over the middle of its top-most pixels: at
       the box's corner the exhaust's number lay on the seat); a candidate that is a part group already
       selects it. Find's chips are of three kinds: the group itself (a check and its name, green), a
       new outline over a group the phrase names (a layers icon, "overlaps"), and one that is mostly a
       part of another kind (an alert icon, last: the same check mark had marked the exhaust can as a
       confirmed spring).
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
       download button and a "Copy share link"; the export line shows the size and the
       time (two items that wrap onto two lines when the info column is narrow; the Download
       button is icon-only in an inspector of 340 px or less, a container query), and a
       second line says when the layers were upsampled from the working resolution instead
       of decomposed at full resolution.
3. **Gallery** — cards of all jobs with 512 px thumbnails (`thumb` of the JobSummary, lazy,
   decoded off the main thread, cropped to the card's aspect), status chips, delete.
4. **How it works** — animated pipeline diagram (five stages light up in sequence),
   one paragraph each, no jargon soup.

Motion: enter transitions (fade + 8 px rise, 240 ms, `cubic-bezier(.2,.8,.2,1)`),
FLIP reorder for lists, skeleton shimmer while loading, count-up numbers, the wipe
handle with spring easing, hover glow on regions, toast notifications bottom-right (in the
studio at the canvas's corner, clear of the inspector: at the page's corner the export toast
covered the export result it announced; above the selection pill while one shows: at 1024 px the
toast after Make group covered the new group's actions; at a phone's width, where the inspector
runs under the canvas, at the top below the header),
`prefers-reduced-motion` respected everywhere. Keyboard: `1–6` layer tabs, `space`
hold to peek original, `⌘/Ctrl+Z` undo mapping change, `Esc` clear selection, `S` Select part
(Enter, Backspace, N and Esc inside it).

Implementation: `js/app.js` (router + boot), `js/api.js` (fetch wrappers + SSE +
abortable render), `js/state.js` (tiny observable store with undo stack for the
mapping), `js/viewer.js` (canvas, layers, hover/select, wipe, zoom, the Select part layer:
`setTool`, `maskSprite`, candidates), `js/panels/*.js`
(groups, palette, mapping, finish, stepper), `js/motion.js` (helpers), `js/toast.js`,
`js/tooltip.js` (one floating tooltip in `<body>` for every `data-tip` element, placed on
hover or keyboard focus, flipped to the side with room and wrapped at 260 px: a
pseudo-element inside the scrolling inspector was clipped to fragments),
`js/util.js` (color math for swatches: hex↔rgb, contrast text color). No frameworks;
the DOM is built with small template helpers. Everything the API returns is rendered
through these panels — no `JSON.stringify` in the UI.

While the backend is being built in parallel, develop against `web/mock/` : a
`mock-server.py` (stdlib `http.server`) that serves the static files and fakes the
API with the sample images and made-up groups/palettes, including SSE progress.
Delete nothing when the real server arrives; the mock stays for UI work.

The studio saves its state (`PUT /state`: mapping, render options, palette) only when it
differs from what the job last had (`markSaved` after a load or a structural edit): opening
a job used to rewrite `job.json` with the same state. At a phone's width (600 px and less) the
header folds (the wordmark keeps its glyph, "How it works" reads "How", the keyboard-shortcut
button goes), the studio toolbar keeps one row (the layer tabs scroll inside it; the photo's
name and the zoom steps go, pinch zooms, Fit and Compare stay), and the home page's hero glow
is clipped at the page edge (`.app-main { overflow-x: clip }`): at 390 px the home page
scrolled sideways to 555 px and the studio to 676 px, now neither does.

Definition of done: opens without console errors on Chrome; Home → drop or sample →
stepper animates → Studio shows layers, hover highlight, palette generation, mapping
drag, live render, export; responsive down to a 1024 px window, and no horizontal scroll at
a 390 px phone width; no layout shift on load; Lighthouse-style basics (labels on inputs,
focus rings, contrast).

## 4. Ownership map (parallel build)

| owner | writes only |
|---|---|
| INT | `recolor/intrinsic/*`, `scripts/dev_intrinsic.py`, `tests/test_intrinsic.py` |
| SEG | `recolor/segmentation/*`, `scripts/dev_segment.py`, `tests/test_segmentation.py`, `tests/test_refine.py`, `tests/test_matting.py`, `tests/test_parts.py`, `tests/test_grouping_parts.py`, `tests/test_junk.py`, `tests/test_subject.py` |
| PAL | `recolor/palette/*`, `recolor/mapping.py`, `scripts/dev_palette.py`, `tests/test_palette.py`, `tests/test_mapping.py` |
| ENG | `recolor/engine.py`, `scripts/dev_render.py`, `tests/test_engine.py` |
| SRV | `recolor/jobs.py`, `recolor/pipeline.py`, `recolor/server/*`, `serve.py`, `tests/test_server.py`, `tests/test_pipeline_idle.py`, `tests/test_pipeline_refine.py`, `tests/test_clickseg.py` |
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
