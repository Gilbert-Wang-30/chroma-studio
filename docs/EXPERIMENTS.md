# Research experiments vs the shipping engine on the two motorcycles

> **Local-only sources.** Paths under `scratch/` (experiment code, per-experiment `README.md` and
> `run_metrics.py`, the halo harness `scratch/halo_eval.py`, renders) are the author's working files.
> `scratch/` is git-ignored, so they are not part of the published repository and the numbers that cite
> them can only be recomputed on the machine they were measured on. The comparison sheets are copied
> into `docs/experiments/` and `docs/comparison/`.

Companion to [RESEARCH.md](RESEARCH.md) (the literature survey). Experiment code, renders and per-experiment
`README.md` / `run_metrics.py` live under the git-ignored `scratch/experiments/`; the two comparison sheets are
copied into `docs/experiments/`.

Date 2026-09-23. Five experiments under `scratch/experiments/<key>/` (each with `README.md`,
`result.json`, `run_metrics.py`, renders and sheets), all measured on the same two jobs: the
Ducati 748 (`data/jobs/4bd1f1789d8f`, paint = group "Red") and the yellow BMW S1000RR
(`data/jobs/d3ed9ee011c9`, paint = the six amber-hue groups), navy target `#123f9e` unless stated.
Every number below is the one an independent verifier reproduced from the saved files with the
project `.venv`; where the verifier corrected an experiment's claim the corrected value is used
and the original is struck through or footnoted. No experiment was refuted outright; the
grouping experiment carries a *major* correction (section 2.2). Nothing tracked was modified
(only `docs/` gained new files).

Comparison sheets (photo, shipping engine, one row per experiment render; full view + the
`metrics.TANK_CROP` / `metrics.BMW_CROP` crop at 2x, labels carry the verified numbers):

* `docs/experiments/ducati_navy_comparison.jpg` (1880 x 5760)
* `docs/experiments/bmw_navy_comparison.jpg` (2000 x 5410, downscaled from 2801 x 7578)

(built by `scratch/experiments/make_compare_sheets.py`; the "current
engine" row is the shipping `recolor/engine.py` render, byte-identical in every experiment folder,
md5 `9f5af716…` / `826168cc…`).

Metric definitions (`scratch/experiments/common/metrics.py`, `scratch/halo_eval.py`):
*hue drift* = area-weighted mean of |median OKLab hue error| over CIELAB-L bins of the repainted
paint / worst bin, degrees, positive = purple-ward; *fragmentation* = share of the main paint's
area held by its largest group (1.0 = one group), paint-locked variant; *halo / bleed / jaggy* =
black-target harness (old-paint pixels left on or beside the repainted parts / repaint > 3 px
outside the label / boundary gradient ratio, > 1.25 = new stair-steps); *leak* = CIELAB chroma of
the luminance-normalised shading inside the paint (median / p90) and the share of image-luminance
variance the albedo explains (0 = flat albedo); *reflection survival* = old-paint-hued pixels
> 20 px outside the paint after / before (1.0 = reflections keep the old colour, FM11).

## 1. Results table (verified numbers; baseline = shipping engine)

| experiment (variant) | hue drift Ducati navy (mean / worst bin) | hue drift BMW navy | fragmentation main share | halo / bleed / jaggy Ducati (black) | halo / bleed / jaggy BMW (black) | leak Ducati (shd chroma med / p90, albedo R2) | leak BMW | reflection survival | fidelity | runtime / VRAM | verifier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline: shipping CIELAB engine + Careaga v2.1 + CIEDE2000 grouping** | 0.69 / 9.43 | 4.79 / 23.01 (reads violet) | Ducati 0.999, BMW 0.911; RX-78 0.531, Sazabi 0.941, red car 0.858, Exia 0.989 | 1884 / 17 / 1.021 (interior dE 0.99) | 2528 / 29 / 1.097 (0.34); 6-job total 11484 / 277 / 1.404 | 10.28 / 33.58, R2 0.37; residual energy 0.14 | 4.56 / 9.38, R2 0.51; residual 0.19 | 1.00 (2720 / 2720 px Ducati; 16837 / 16837 BMW, mostly the un-grouped nose) | identity 0; 60/60 unit tests | render 4.77 / 5.47 ms preview, 9.45 / 10.16 ms work res (Ducati / BMW); Careaga 0.49 s per image, 4.8 GB | – |
| **oklab** (`engine_oklab.py`, OKLab chroma + CIELAB-like toe on OK L; needs `LIGHTNESS_MODE="toe"`, the file's default `"ok"` crushes blacks) | 0.43 / **20.97** (worst bin 2.2x worse: top 1 % highlights go cyan-ward, −21 vs −9 deg); pastel 1.19 / 2.85 → **2.65 / 6.30 (worse)** | **2.07 / 7.14** (reads blue); pastel 3.11 / 8.29 → 2.23 / 7.88 | unchanged (engine only) | 1881 / 17 / 1.025 (1.32) | 2531 / 31 / 1.101 (0.48); 6-job 11473 / 282 / 1.400 | unchanged | unchanged | 1.00 unchanged | identity 0; black repaint L median BMW 15.2 → 16.7, Ducati 0.5 → 2.0 (pure OK L: 6.3 / 0.1, rejected); 58 pass / 1 marginal fail (chroma 9.16 vs < 9.0) + 1 test re-asserted in OK units | +0.2–0.9 ms per render (4.94 / 5.66 preview, 10.35 / 11.05 work res), ~~"same or better"~~; 0.54 GB per renderer; experiment 1440 s, 1.2 GB | minor; verdict *improves* (BMW-led) stands |
| **grouping** (`grouplib.make_d_blend`: shading-invariant CIEDE2000, kC = 2, lightness weight 0.15 for chromatic pairs, shipped distance for greys, thr 10; + decal split on the Ducati) | 0.70 / 10.16 (with decal split) | 3.62 / 20.28 | **RX-78 0.531 → 1.000, Sazabi 0.941 → 0.993, red car 0.858 → 0.961**, BMW 0.911 → 0.913, Ducati 0.999, Exia 0.989 | 2088 / **100** / 1.018 (0.85) — decal rim now an unpainted neighbour | 2477 / 33 / 1.089 (0.36); RX-78 472 / 33 / 1.026 → 894 / 42 / 1.036 | unchanged | unchanged | 1.00 unchanged | identity 0; RX-78 navy hue drift 1.56 / 56.2 → 6.98 / 24.9, per-pixel mean \|err\| **5.5 → 9.7 deg** (merged dark pieces +10…+15 deg purple, render is consistent but off-navy); BMW **3 wrong merges + brake reservoir painted** ~~(2 + 1 debatable)~~; ~~"greys group exactly as today"~~ (Sazabi, BMW r35, red-car r138, Exia r149 move) | analysis-time only (agglomeration; `run_metrics` 16 s); experiment ~21 min ~~6000 s~~, 1.48 GB; DINOv2 / SAM features: negative at every weight | **major** (mechanism and safety claims corrected; numbers reproduce) |
| **matting** (`MatteRenderer`, ViTMatte-small, 6 px band, max with guided filter, then the engine ramp) | unchanged 0.69 / 9.43 | unchanged 4.79 / 23.0 | unchanged | **1627** (−14 %) / **187** / 1.017 (0.99); navy 1269 / 16 → 1146 / 135 | **1177** (−53 %) / **616** / 1.064 (0.34); navy 1396 / 97 → 728 / 690 | unchanged | unchanged | 1.00 unchanged | identity 0; bleed split by what the photo shows: BMW 457 old-paint / 159 other (looser test 533 / 89), Ducati 18 / 169 (39 / 150) | +31 ms per new mapping set (small; base 48 ms), first matte 151 ms ~~190~~, 2.1 GB transient (base 3.7 GB), `transformers` + 100 MB model; experiment 960 s | minor |
| **intrinsic** (`mg_l1536_gs`: Marigold-IID lighting at 1536 px, fp16, 4 steps, guided-filter edge snap, shading rescaled, group colours re-measured) | **2.5 / 14.2** (worse) | **1.2 / 3.0** (magenta highlight cast gone) | unchanged (Careaga groups kept) | 1876 / 6 / **0.87** (softer edges) (0.5) | 2467 / 31 / 1.07 (0.3) | **4.4 / 6.5, R2 0.01**; residual 0.32 (2.3x) | 1.3 / 4.9, R2 0.72 (albedo only modestly flatter: L std 8.0 → 5.7 at 12 px erosion); residual 0.36 | 1.00 unchanged | identity 3–4 / 255 (float16 dense residual); navy L median Ducati 21.4 → 29.2 (target 29.8, closer), BMW 34.3 → 43.9 (royal blue, 15.5 L off); black Ducati tank loses its gloss (structure r 0.68 → 0.46), BMW black L 15.2 → 22.1 (grey); interior detail 0.76 / 0.87 of the photo vs 0.87 / 1.21 (decals blur, vanish on black) | 0.9 s per image (Careaga 0.49), 6.0 GB alloc / 7.4 GB nvidia-smi, 2 × 2.6 GB checkpoints (openrail++), `diffusers` venv; experiment 1440 s | minor |
| intrinsic follow-up **`mg_hyb`** (Careaga layers, Marigold shading chromaticity only) | **0.4 / 3.4** (best seen) | 2.1 / 20.9 | unchanged | 1882 / 22 / 1.05 | 2567 / 29 / 1.08 | 4.4 / 7.0, R2 0.37 | 1.4 / 5.2, R2 0.51 | 1.00 | identity 4 / 3; navy L 24.8 | +0.9 s per job, same model cost | reproduced |
| **rerender** (RGB→X→RGB, Zeng 2024, engine albedo edit, seed 0) | 13.0 / 27.2 | 4.7 / 5.4 | unchanged | not run (not a `render(bundle, mapping, options)` callable) | not run | unchanged | unchanged | **0.245 (1 % navy, 99 % deleted)**; BMW 0.08 (66 % navy = the un-grouped nose) | **PSNR far from bike 3.8 dB** (unedited re-render 6.8 dB overall / 5.1 dB far, LPIPS 0.54–0.62; full res 4.7 dB; BMW 16.3 / 22.4 far / 13.1 on the bike; VAE cap 25.0 / 28.4 dB); paint dE 35.5 vs engine | rgb→x 8.5 s + x→rgb 4.0 s per run, 6.7 GB, 13.8 GB weights; experiment 4200 s | minor |
| rerender: X→RGB **inpainting** (mask = paint + reflecting px, composited) | 15.4 / 18.7 | 11.2 / 13.9 | – | – | – | – | – | 0.14 (5 % navy, 46 % neutral); BMW 0.036 (63 % navy) | paint dE 27.4 (flat pale lavender, decal lost); chrome → teal/grey blotches | 4.3 s per run | – |
| rerender: **no-diffusion control** (Lab hue rotation of old-paint-hued pixels outside the paint) | 0.7 / 9.4 (paint untouched) | 4.8 / 23.0 | – | not run | not run | – | – | 0.00 **by construction** (rotates exactly the pixels the metric counts): 37 % of 2720 px turn navy, 63 % drop below chroma 20; BMW 90 % navy | dE 0 inside paint; 37.2 / 26.5 dB vs engine render outside the paint; 0.47 % / 1.7 % of pixels changed | two lines of numpy, no model | recommendation stands, metric is tautological |

Legend: bold = the number that moved and matters; ~~struck~~ = experiment claim corrected by the
verifier. "unchanged" = identical to baseline by construction (the experiment does not touch that
stage). Pastel target `#7fb2ff` and black `#000000` were measured for the oklab experiment only.

## 2. Per experiment

### 2.1 `oklab` — colour math in OKLab/OKLCH instead of CIELAB (FM6 hue drift, FM5 darkness)

**What was tried.** `recolor/engine.py` copied to `engine_oklab.py` with the chroma rotation /
scaling, anchored lightness map, gamut compression (chroma binary search at fixed OK L and hue),
bounce retint and residual tint moved to OKLab (torch M1 / cbrt / M2, identity bit-exact,
round trip 2e-6); CIELAB thresholds translated with a measured scale (OK chroma ≈ CIELAB / 350
near neutral). Four variants: straight port (`ok`), CIELAB-like toe on OK L (`toe`, the proposal),
retint on the measured per-group leak direction (`toe_leak`), OKLCH chroma relative to the gamut
boundary (`toe_gamut`). Measured on navy / pastel / black for both bikes, halo harness on all six
analysed jobs, shipping unit tests run against the copy, live timing and VRAM.

**What happened.** The user-visible error is fixed on the BMW: the shipping navy renders violet,
the OKLab navy blue (4.79 / 23.0 → 2.07 / 7.1 deg); BMW pastel also improves (3.1 / 8.3 → 2.2 / 7.9).
On the Ducati the experiment's own metric does not improve: navy mean 0.69 → 0.43 but the worst
bin doubles (9.4 → 21.0 deg, the top ~1 % of paint — tank ridge highlight and decal edges — drifts
cyan-ward instead of purple-ward and is slightly greener at native resolution), and the pastel gets
worse on both numbers (1.19 / 2.85 → 2.65 / 6.30, a 2–4 deg cyan bias). Both are traced to the
bounce retint: the paint's leak into the shading sits 22 deg warmer than the paint in OKLab vs
17 deg in CIELAB, so the paint-direction projection removes a little less of it. Pure OK L crushes
black repaints (BMW interior L median 15.2 → 6.3, floor 29 % → 41 %); the toe restores them
(16.7 / 28 %). Halo / bleed / jaggy / identity unchanged on all six jobs (11484 → 11473 px halo).
Measured in CIELAB hue the ranking flips on the bright bins of both bikes (Ducati 65+ bin
−24.4 → −36.5, BMW +1.0 → −10.5): the two hue systems disagree along the blue lightness axis and the
sheets, not the numbers, settle it (`compare_bmw.jpg` rows 1 vs 2). `toe_leak` passes all 59 tests
and is the best variant on the BMW (1.9 / 6.8, pastel 1.4 / 4.6) but turns the Ducati pastel shadows
+7–9 deg purple; `toe_gamut` changes nothing in hue and pushes 34–37 % of a navy render onto the
gamut boundary.

**Verifier corrections applied.** `engine_oklab.py` ships with `LIGHTNESS_MODE = "ok"` (the
rejected variant) — every reported `oklab_toe` number depends on `run_metrics.configure()` flipping
it; OKLab is consistently 0.2–0.9 ms *slower* per render (4.94 / 5.66 vs 4.77 / 5.47 ms preview;
10.35 / 11.05 vs 9.45 / 10.16 ms work res), not "the same or better"; the Ducati is neutral to
marginally worse, the "improves" verdict rests on the BMW. All numbers reproduce exactly; identity
vs photo 0 for both engines.

**Verdict.** Improves (BMW-led; Ducati highlights marginally worse, 1 % of the paint). Severity of
corrections: minor.

**What it would take to ship.** (1) Port the OKLab chroma plane and the toe lightness rule into
`recolor/engine.py` with `toe` as the only lightness mode (the finished port is the reference; the
CIELAB helpers go once the tests are ported); (2) re-assert `test_hue_turn_fades_out_for_neutral_sources`
in OK units and either move the `test_light_repaint_of_saturated_group_is_neutral` bound from 9.0 to
9.5 or, better, fix the cause by projecting the bounce retint on the measured per-group leak
direction with the rotation origin handled per pixel (the `toe_leak` variant minus its light-target
regression); (3) re-tune the three translated confidence thresholds (hue confidence 2..8, neutral
cut 6, well-lit L 25) on the other four jobs — they were scaled on the two bikes only;
(4) accept the +0.2–0.9 ms. One to two days; no new dependency; VRAM unchanged.

### 2.2 `grouping` — shading-invariant and feature-based regrouping (FM1)

**What was tried.** The shipped area-weighted centroid-linkage agglomeration rebuilt with a
pluggable distance (reproduces the shipped groups of all six jobs exactly). (a1) uniformly
down-weighted lightness — merges black bodywork with graphite engines, rejected; (a2) colours
rescaled to L = 50 (exact for a grey multiplicative leak above CIELAB's toe) + chroma-gated
lightness term, threshold sweep 8–16 (10 is the last safe point; 12 merges the Ducati caliper with
the wheel, 16 the BMW fork with the paint); (a3) CIEDE2000 parametric kC = 2 for chromatic pairs
(catches highlight-washed albedo: red-car bonnet, BMW tail underside); (a4) `blend`: shipped
distance for pairs with a near-neutral member, the invariant kC = 2 distance for chromatic pairs.
(b) DINOv2 ViT-B/14 and (c) SAM 2.1 encoder per-region mean features in a joint distance (weights
0.05–0.4). Plus a pixel-level decal outlier split of the Ducati paint regions, and navy / black
renders of BMW, RX-78 and Ducati with baseline vs regrouped groups through the real `Renderer`.

**What happened.** The `blend` rule fixes fragmentation where the split comes from shading leak or
highlight wash: RX-78 0.531 → 1.000, Sazabi 0.941 → 0.993, red car 0.858 → 0.961; Ducati and Exia
were never split; BMW 0.911 → 0.913. DINO / SAM region features are a clean negative: they encode
which part a region is, not its material (the BMW's own tail panel is farther from the fairing,
1−cos 0.57, than the gold fork tube 0.48 or the mirror stalk 0.49), so any useful weight fragments
the paint. The decal split isolates the tank (1042 px) and fairing (5048 px) DUCATI lettering as
their own regions, grouped with the silver parts, at the cost of a newly exposed rim (Ducati bleed
17 → 100, halo 1884 → 2088; the render is cleaner white lettering, `compare_ducati.jpg` row 3).
On the two motorcycles the regrouping itself is a wash: the Ducati is unchanged, and on the BMW it
merges the mirror stalk (black plastic whose albedo is dark yellow from interreflection), the
shadowed panel under the mirror (colour-identical to the gold fork tube after normalisation) and
a black side panel into the wrong groups, and the brake-fluid reservoir ends up painted navy — a
new visible artifact on a non-paint part.

**Verifier corrections applied (major).** The RX-78 render is *consistent but not navy*: the
README attributes the drift rise (1.56 → 6.98 deg) to the bright shield landing in high-L bins, but
measured per piece the shield (the anchor, L 54) drifts +0.1 deg in both renders while the newly
merged dark pieces (chest, feet, chin, L 28–41) go +10…+15 deg purple under the brighter anchor;
per-pixel mean |OKLab hue error| rises 5.5 → 9.7 deg over ~half the painted area, and the quoted
"worst bin 56 → 25 deg" rests on a 392-px bin that disappears. "Greys group exactly as today" is
false in detail (Sazabi r26/r65/r158/r165/r184 regroup, r175+r189 red plastic → Graphite and no
longer recolourable; BMW r35 black panel → fork-tube group and painted; red-car r138; Exia r149).
BMW wrong merges are 3 plus the painted reservoir, not "2 + 1 debatable". The decal split carves
six regions (6897 px), not two: four are specular / shadow blobs. The red-car background Ford
joins the main car's paint group. `trusted_list gained facebookresearch_dinov2` is false (weights and
hub code are present). `runtime_s 6000` is unsupported (~21 min of file activity). All
fragmentation, harness and hue-drift numbers reproduce exactly.

**Verdict.** Mixed. Real FM1 gain on three of the four non-motorcycle jobs, nothing on the Ducati,
net negative on the BMW; and on its own it swaps seams for a hue error on merged pieces, because
the engine repaints a wider lightness range under one anchor.

**What it would take to ship.** (1) `make_d_blend` as the linkage in `grouping.cluster_colors`
at threshold 10, *after* the OKLab lightness work lands (or with a per-piece anchor so a merged dark
piece is not repainted under a bright anchor — the RX-78 purple shift must be re-measured per
piece, not per bin, before this ships); (2) keep the grey partition literally the shipped one
(apply the blend distance only when both cluster centroids stay chromatic through the merge
sequence, or run the shipped grouping for neutrals first) so the Sazabi / BMW grey moves cannot
happen; (3) a guard against the BMW-type mix-ups — colour cannot separate them, so either a
non-colour cue (material, section 3) or ship it as a "merge same paint" suggestion the user
confirms in the groups panel rather than a silent default; (4) the decal split only together with
the colour-guided label refinement so the exposed decal rim does not bleed (bleed 17 → 100 px
today). One to two days for (1)+(2); (3) is open.

### 2.3 `matting` — alpha-matte coverage at part boundaries (FM2 halo / bleed / jaggy)

**What was tried.** ViTMatte small and base (fp16, whole 1536 px image, own venv with
`transformers` 5.17) on a trimap from the repainted-group mask (band 4 / 6 / 8 px, thin components
rescued); `MatteRenderer` subclasses the shipping `Renderer` and overrides only `_coverage`
(matte replaces / max / mean with the colour guided filter in the band, `m >= hard label` kept,
with and without the engine's outward ramp); alpha-compositing variants (`fill`, wrong by
construction; `delta` `A' = A + a·(f(P) − P)`, the correct unmixing); controls (guided filter
without ramp; shipping engine with feather 3.0 and 4.5). 22 variants × 2 bikes through the black
and navy harness, hue drift, reflection survival, coverage statistics, bleed split by photo content,
3x boundary crops.

**What happened.** A raw matte in the band makes halo *worse* (3100–3300 px) because the engine
keeps (1−m) of the old paint at mixed pixels — the outward ramp is what removes ~1500 halo px per
bike today. Matte + ramp: halo 1884 → 1627 (−14 %) on the Ducati and 2528 → 1177 (−53 %) on the
BMW, navy 1269 → 1146 / 1396 → 728, with jaggy, interior, identity, hue drift and reflections
unchanged; a wider ramp cannot reproduce it (feather 3.0 buys −10 % halo for 2434 / 2694 bleed px).
Cost: bleed 17 → 187 and 29 → 616 px — on the BMW 457 (533 with a looser test) of the 616 are
yellow-hued pixels 4–9 px outside SAM's label, i.e. paint the label missed and now repainted
correctly; on the Ducati most of the 187 are dark-red edge pixels and a few px onto the fork and
headlight rim. ViTMatte's alpha is nearly binary here (17 % soft pixels in the band), so the correct
`delta` compositing fails the jaggy threshold (1.31 > 1.25); ViTMatte-base gives the same numbers
at 1.7x the VRAM. Thin painted parts get alpha < 0.5 on 791 / 460 px and survive only through the
`m >= hard label` rule. Visually the 3x crops are nearly indistinguishable, except at the BMW's
lower side-fairing vents / belly-pan seam where the baseline leaves a clear 3–5 px yellow sliver
that the matte removes (the README undersells this).

**Verifier corrections applied.** First-matte warm-up 151 ms (README: ~190); ViTMatte-base load
8.7 s including download, 0.36 s for small; `logs/vitmatte_vram.json` is empty (VRAM numbers
survive in `logs/sweep2.log`: small 2096 MB, base 3663 MB); the Ducati front-wheel crops show the
gold Brembo caliper painted black in *both* engines (a pre-existing grouping issue, not a matte
artifact). Everything else reproduces byte-identically; changed pixels lie within −3…+9 px of the
label boundary on all four renders.

**Verdict.** Mixed — the halo gain is real but small to the eye, and not worth a per-render model
(`transformers`, 100 MB, ~2 GB transient VRAM, 30–150 ms per new mapping set).

**What it would take to ship.** Not as a coverage change. The experiment's real finding is that
the *label boundary* sits a few px inside the paint on the BMW; use the matte once at analysis
time to snap `labels.npy` / `group_map.npy` (`refine_labels_with_guide` with the alpha as guide, or
the colour guided filter already in the engine) and keep the shipping engine untouched — same halo
reduction, no per-render model, and it is the fix the decal split (2.2) needs. Keep the `delta`
compositing formula for whenever a trustworthy soft alpha exists (a 1-px anti-aliased label edge);
do not use ViTMatte-base. One day for the label-snapping prototype, measured with the same harness.

### 2.4 `intrinsic` — Marigold-IID vs Careaga v2.1 (FM3 leak, FM5 highlights)

**What was tried.** Own venv (`diffusers` 0.40, `transformers` 5.17); both prs-eth Marigold-IID
v1-1 checkpoints (lighting: albedo + diffuse shading + residual; appearance: albedo + material),
fp16, 4 DDIM steps, at 768 px (default), 768 with 4-sample ensemble, native 1536 px, 1536 with
ensemble 3 and with 10 steps. Layers made engine-consistent (shading rescaled by global least
squares, residual = photo − albedo·shading, exact in float32), 768 outputs upsampled with the
photo-guided filter, full-res layers edge-snapped against the photo (final `mg_l1536_gs`). A
source-colour confound found and fixed (group `albedo_lab` re-measured on each variant's albedo;
Marigold's Ducati red is Lab 34/46/36 vs Careaga's 42/66/54, which otherwise gave a constant
+9…+13 deg hue error). Mixes: Marigold albedo + Careaga shading, appearance albedo, low-frequency
shading correction, global albedo scale calibration, and `mg_hyb` (Careaga layers + Marigold's
shading chromaticity only). Lightness and highlight-structure metrics added after the sheets
showed the navy coming out lighter and the black tank losing its gloss.

**What happened.** The decomposition is genuinely cleaner on the Ducati (shading chroma 10.3 / 33.6
→ 4.4 / 6.5, albedo R2 0.37 → 0.01, robust to mask erosion) and its BMW shading is more neutral
(4.6 / 9.4 → 1.3 / 4.9), which removes the BMW's magenta highlight cast (hue drift 4.8 / 23.0 →
1.2 / 3.0). But: the Ducati hue drift gets worse (0.7 / 9.4 → 2.5 / 14.2); layers are blurry at part
boundaries (VAE decoder) and even after the edge snap the paint interior loses high-frequency
detail (0.76 / 0.87 of the photo vs 0.87 / 1.21 for Careaga) — the DUCATI decals go soft and
grey-blue on navy and vanish on black; 2–3x more of the paint's energy sits in a dense,
paint-coloured residual (0.14 / 0.19 → 0.32 / 0.36) that the engine discards on dark targets, so the
black Ducati tank becomes a flat slab (structure r 0.68 → 0.46) and the black BMW fairing a flat
grey (L 15.2 → 22.1); the absolute albedo scale differs per material, so the same navy lands
~8–10 L units lighter (Ducati 21.4 → 29.2, which is actually *closer* to the target's 29.8; BMW
34.3 → 43.9, royal blue, 15.5 L off) and no global calibration fixes it. Ensembles and more steps
change nothing. The unsnapped 1536 variant reads better on the numbers (2.5 / 9.6) but shows a
mottled magenta cast over the Ducati fairing that the per-bin median metric does not see. The
cheap hybrid `mg_hyb` is the best Ducati hue result of any experiment (0.4 / 3.4 mean / worst) at
zero sharpness cost and does nothing for the BMW (its drift lives in Careaga's albedo).

**Verifier corrections applied.** Interior detail loss under-reported (numbers above); "paint
albedo nearly flat" holds for the Ducati only (BMW L std 8.0 → 5.7 at 12 px erosion, R2 above
Careaga's until 24 px erosion); lightness framing is one-sided (Marigold is closer to target on the
Ducati, off on the BMW); BMW black-target L 22 omitted from the verdict; minor README
inconsistencies (fitted scale a = 3.9 vs 4.07; trial-log vs verdict variant). All headline numbers
reproduce exactly.

**Verdict.** Mixed. Not a drop-in replacement for Careaga.

**What it would take to ship.** Do not swap the decomposition. Two follow-ups: (1) `mg_hyb` —
graft Marigold's neutral shading chromaticity onto Careaga's layers (or use it to validate the
engine's `_light_reference` illuminant estimate); cost +0.9 s per job, 6.0 GB during analysis, one
2.6 GB openrail++ checkpoint and `diffusers` in the venv — run it on the four other jobs first, adopt
if they agree; (2) for the BMW-type failure (highlights baked into Careaga's albedo, R2 0.51),
flatten Careaga's albedo inside each paint group toward its median with Marigold's albedo as the
regression target, keeping Careaga's edges. If Marigold were ever adopted wholesale it needs native
1536 px, single 4-step sample, edge snap, re-segmentation on its albedo, float32 residual storage
and a lightness rule for the albedo scale, plus the licence review. About a week.

### 2.5 `rerender` — diffusion re-rendering for global consistency (FM11 reflections, FM5 fidelity)

**What was tried.** RGB↔X (Zeng et al. 2024; code from `zheng95z/rgbx`, one import patch for
diffusers 0.40; 13.8 GB of weights) in its own venv. Step 1 rgb→x on both photos at 1152×768
(and the Ducati at 1536×1024). Step 2 the fidelity ceiling: x→rgb from the *unedited* channels,
9 configurations (prompts, seeds, no-CFG, guidance, 100 steps) + full res, PSNR overall / far from
the bike / on the bike, LPIPS, VAE round-trip cap, and whether the re-render even reproduces the red
reflection pixels. Step 3 paint edits (engine albedo pasted; multiplicative ratio — failed, turns
the bike green; the engine's Lab shift rule; hue rotation of the reflecting pixels in irradiance /
albedo), 3 seeds. Step 4 the X→RGB inpainting UNet (masks: paint; paint + reflecting pixels; whole
bike + floor band), composited onto the engine render. Two non-diffusion references: a masked
blend, and a control that hue-rotates old-paint-hued pixels outside the repainted groups in Lab
with the engine's theta and chroma scale.

**What happened.** RGB↔X is an interior-scene prior, not a renderer of this photograph: fed its
own unedited channels it re-lights the Ducati's black studio as a concrete room (PSNR 6.8 dB
overall, 5.1 dB far from the bike, 4.7 dB at full res; LPIPS 0.54–0.62) and redesigns gold wheels,
chrome and exhaust as grey plastic; the BMW keeps its light backdrop (22–24 dB far) but the bike
is 13 dB; the VAE alone caps everything at 25–28 dB. Decisively, it never synthesises this scene's
interreflections: 93–99 % of the 2720 red reflection pixels come back neutral even unedited, so
every "survival" drop is deletion, not recolouring (≤ 5 % turn navy). The paint itself comes out
5–13 deg off in hue and dE 35–39 from the engine render. The inpainting composite keeps untouched
pixels but paints the tank a flat pale lavender (dE 20–28, decals destroyed) and turns the fork
tube, caliper and mirror into teal / grey blotches (survival 0.14). The masked blend passes the
metric by pasting re-lit speckles into the dark fork. The no-diffusion control reaches survival
0.00 on both bikes at dE 0 inside the paint and 37.2 / 26.5 dB vs the engine render outside it,
touching 0.47 % (Ducati: fork tubes, tank / fairing seams, tail reflection) and 1.7 % (BMW:
essentially the un-grouped nose and upper cowl) of the pixels, and looks clean in the zooms.

**Verifier corrections applied.** The control's survival 0.00 is tautological (it rotates exactly
the pixels the metric counts); the meaningful numbers are 37 % navy / 63 % neutral (Ducati) and
90 % navy (BMW), and it was never run through the halo harness, so its cost on edges and on the
gold wheels (hue 68 deg, 30 deg from the Ducati red — the feathered variant let partial rotation
of gold pass through red) is unmeasured. Range overstatements: unedited neutral fraction is
92.9–98.8 % (full res keeps 12.8 % red), BMW hue-rotated variants 0.080–0.205, one ratio variant
at 36.9 deg. All numbers reproduce byte-identically.

**Verdict.** No gain for diffusion re-rendering (or its inpainting variant), on either bike.

**What it would take to ship (the by-product, not the method).** An FM11 step inside the engine:
find old-paint-hued pixels outside the repainted groups (chroma > 20, hue within ~30 deg of the
source albedo, not in a locked group) and apply the same Lab/OKLab rotation and chroma scale as the
paint, weighted by a *soft* gate on chroma, hue distance, distance to the repainted parts and a
hue margin that protects the gold wheels (hard mask, not a feathered RGB blend). Sweep the gate
(chroma 15 / 20 / 25, window 20 / 30 / 40 deg, falloff 20 / 40 / 80 px, gold margin 15 / 20 deg),
judge by the halo harness, PSNR outside the paint and dE inside the gold groups (< 2), not by
`reflection_survival`. On the BMW fix the grouping miss first (nose and upper cowl are not in the
amber groups); it dominates that job's FM11 count. One to two days.

## 3. What should go into the product first

Ranked by verified gain on the two motorcycles per unit of cost and risk, with the other four
analysed jobs as a tie-breaker.

1. **OKLab chroma plane + toe lightness in `recolor/engine.py`** (`oklab`). The only experiment
   that changes what a user sees on a real repaint of the BMW (violet → blue, 23 → 7 deg worst
   bin) with zero cost in halo, coverage, darkness, dependencies or VRAM and +0.2–0.9 ms per
   render; the finished port exists. Risks are small and known: the Ducati's top 1 % of highlight
   pixels go slightly cyan, one unit-test bound moves, thresholds were scaled on two photos. Ship
   first; follow with the leak-direction retint once its per-pixel rotation origin is solved.
2. **In-engine gated hue rotation of out-of-group old-paint pixels** (the `rerender` control).
   Two lines of colour math that visibly recolour the Ducati's fork / caliper / floor reflections
   (37 % navy, the rest dimmed) with dE 0 inside the paint and 37 dB outside — but it has not been
   through the halo harness and needs the soft gate designed and swept before it is safe on gold
   wheels. Ship second, after that sweep; on the BMW it mostly paints the un-grouped nose, which
   is really item 4's job.
3. **Label snapping at analysis time** (the finding of `matting`, not its method). SAM's label
   sits a few px inside the BMW's paint; snapping `labels.npy` / `group_map.npy` to a colour-aware
   edge (ViTMatte alpha once per job, or the engine's own colour guided filter) is where the
   −53 % halo lives, with no per-render model. Prototype and measure with the harness before
   committing to the `transformers` dependency; this also unblocks the decal split.
4. **Shading-invariant `blend` linkage in `grouping.cluster_colors`** (`grouping`). Large,
   real gains on the RX-78 / Sazabi / red car, nothing on the Ducati, net negative on the BMW
   (three wrong merges plus a painted brake reservoir), and the merged pieces pick up a purple
   hue error under a single anchor. Ship only after item 1 (or with per-piece anchors), with the
   grey partition pinned to the shipped one and the BMW-type mix-ups either guarded by a
   material cue or surfaced as a confirmable "merge same paint" suggestion. The decal split goes
   with item 3.
5. **Marigold shading-chromaticity graft (`mg_hyb`)** (`intrinsic`). Best Ducati hue number of
   the whole study (0.7 / 9.4 → 0.4 / 3.4) at no sharpness cost, but it buys nothing on the BMW and
   costs +0.9 s per job, 6 GB during analysis, a 2.6 GB openrail++ checkpoint and `diffusers`.
   Run on the other four jobs first; adopt only if they agree. Do not swap Careaga for Marigold
   wholesale.

Not to pursue, measured: diffusion re-rendering or inpainting for reflections or as a refinement
(deletes reflections, re-lights the scene, 4–7 dB on the Ducati); ViTMatte as a per-render
coverage change (halo gain the eye barely sees for bleed, a dependency and 2 GB); ViTMatte-base;
DINOv2 / SAM 2.1 region features for grouping (fragment every paint by part); OKLCH chroma
scaling at fixed L (pastel texture loss is a lightness-direction gamut problem needing a cusp-aware
mapping); pure OK L without the toe (crushes black repaints).

Open measurement gaps that the ranking depends on: the control of item 2 has no harness numbers;
item 4's RX-78 hue error must be re-measured per piece after item 1 lands; items 1 and 5 were
measured for hue and darkness on the two bikes only (the four other jobs saw only the black-target
harness); `reflection_survival` is tautological for any method that edits exactly the pixels it
counts and saturates at 1.00 for anything that edits only inside the paint.
