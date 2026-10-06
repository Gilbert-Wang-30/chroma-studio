# Part-aware grouping: what was wrong, what was measured, what shipped

> **Local-only sources.** Paths under `scratch/` are the author's working files: the reference sets, `partkit` and the by-eye junk catalogue (`scratch/groupexp/ref/`), the three experiment lanes with their `README.md`, `result.json` and `run_metrics.py` (`scratch/groupexp/exp/<key>/`), the integration's measurements (`scratch/groupexp/integ7/`), the checks (`scratch/integration5/checks/`) and the sheet builder (`scratch/groupexp/partsheets/`). `scratch/` is git-ignored, so numbers that cite these files can only be recomputed on the machine where they were measured. The human comparison sheets are copied into `docs/comparison/parts/`.

This report is a companion to [SEGMENTATION.md](SEGMENTATION.md) (the round-3 segmentation research) and [EXPERIMENTS.md](EXPERIMENTS.md) (the engine experiments). The contract for what shipped is [ARCHITECTURE.md](ARCHITECTURE.md) §3.2 and §3.8.

Date: 2026-09-30. Code: the working tree after the integration, uncommitted on top of `8dc9826`. `.venv/bin/python -m pytest tests/` passes 421 tests in about 9.6 s.

The round had six parts:
- a reference set of the parts people personalise, on ten sample photos, with its own metrics (`partkit`) and a by-eye junk catalogue;
- two literature surveys: open-vocabulary part detectors, and shadow, lighting and region-merging cues;
- three experiment lanes: `parts`, `junk` and `stack`;
- an integration into `recolor/`, fixed over several check rounds;
- three checks of the integration's last round: numbers, code review, and UI in headless Chromium;
- three visual judges scoring human comparison sheets.

How the numbers were verified:
- Every lane number below is one that an independent verifier reproduced from the saved outputs with the project `.venv`. Where a verifier corrected a claim, the corrected reading is used.
- No lane was refuted. All three carry minor corrections.
- The numbers check reproduced the integration's totals from two fresh analyses of its own. Numbers that only the integration measured are marked "(integration)".
- The numbers check and the UI check each failed on one major defect: the Ducati's brake disc and the Torana's rear rim. Both are still open (section 6).
- The lanes and the checks edited no tracked file. `data/jobs` and the six fixture jobs were only read, and their md5s are unchanged.

**The owner's brief (2026-09-29), in substance.** The grouping has two faults:
- **(A) Junk groups:** a shadow with a slight colour cast gets a group of its own.
- **(B) Missed parts:** the parts people personalise are grouped with unrelated parts of the same colour, such as a motorbike's shock spring or a car's brake calipers.

The principle asked for: a part people personalise (springs, calipers, rims, levers, mirrors, exhaust tips, seats, grilles, badges) is its own group whatever its colour. A colour-only lighting variation of one surface never is.

**Outcome.**
- **Parts.** Detected parts are now groups of their own, named after the part. On the ten photos, isolated parts went from 6 to 18 of 194. At kind level, where one "Mirrors" row holding both mirrors counts for both, they went from 2 to 27 of 133. The segmentation ceiling rose too: region isolation went from 0.490 to 0.500-0.505.
- **What is found.** Shock springs, grips, seats, sprockets, exhausts, tyres, rims, grilles, mirrors, bumpers, door handles, logos, bottles and pedals. A second look at every wheel also finds 6 of the 7 visible brake calipers in the twelve-photo set.
- **Junk.** Tiny lighting-variant groups are folded into their surface. By-eye junk fell from 16 to 10 groups and from severity 25 to 15, including the owner's colour-shifted flare shadow. 3 of 17 real small groups are counted as lost, one of them on purpose.
- **Judges.** All three judges rate the result better than the same analysis without these two mechanisms: a mean of 6.7 against 2.5 out of 10.
- **Still wrong.** The Ducati's brake disc is a speckled mask. Rims do not consistently hold their lip and spokes. No lever and no bicycle brake is found. Sneaker panels and Gundam armour are not detected. Reflections of other objects keep groups of their own.

## 1. The two complaints and their mechanism in code

### 1.1 Missed parts (B)

**The code before this round.** Grouping was colour only:
- `grouping.group_regions` calls `cluster_colors`: area-weighted centroid linkage in CIEDE2000 on each region's median Careaga albedo, at one fixed threshold (`delta_e` 10). It has no notion of adjacency, boundary, region size or part identity.
- `absorb_lit` then joins a chromatic group to a larger anchor when their lightness-normalised albedo (within 8) and photo colour (within 7) agree. This also joins whole parts that happen to be painted one colour.

The regions stage did name parts, and then dropped the names:
- Florence-2 grounded one caption of 12 phrases: tire, wheel rim, brake disc, fork, exhaust pipe, muffler, headlight, turn signal, mirror, logo, emblem, spoke.
- `smallparts.part_masks` prompted SAM with each box.
- `hierarchy._stamp_extras` kept a mask only where it split a region into halves differing by `NAMED_DE` 8 in albedo or `NAMED_DL` 12 in photo lightness. It recorded only `{'source': 'named'}`. `Extra.labels` stopped there, so no phrase reached `Region`, `regions.json` or the grouping.

So a spring the same colour as its frame was either never stamped (it fails the split test) or stamped and then merged back by colour. The only part guard was `refine.isolate_parts`: it takes regions of source `'part'` (the off-colour pockets of `recover_parts`) out of the paint and locks them.

**Measured on the reference set** (section 2):
- 6 of the 194 must-be-separate parts are isolated (0.031).
- Region isolation is 0.490: 95 parts have a region of their own and the grouping loses 89 of them; 99 never get a region.
- Of the 188 missed parts: 82 have no region of their own; 15 have one but are split over groups; 76 are lost to `cluster_colors`' colour-only agglomeration; 12 to `absorb_lit`; 3 to later steps.

The owner's examples and the pattern behind them:

- **Ducati shock spring** (3,136 px, its own region at purity 0.92). It sits in "Gold" (7,950 px) with two frame-tube regions. The lit gold tube is dE 6.6 from the spring's yellow (L 69 b 61 against L 77 b 51), so they cluster. Then `materials.absorb_highlights` moved a frame highlight region into the same group (25 % highlight pixels, the rest within 18 deg of the hue). Painting the spring paints the frame.
- **Ducati front caliper** (the gold Brembo, its own region, purity 0.90). It sits in "Khaki" (57,494 px) with the front rim, the disc and the shock body; their region medians are dE 4.3-7.6 apart. The disc and rim share one region (disc 49 %, rim 37 %, tyre 10 %), because `wheels.split_wheel` puts everything inside the lip on the rim side.
- **BMW rear caliper.** It sits in "Graphite" with the front disc, the fork slider, the front rim and the swingarm (dE 5.7-6.9). The BMW's gold front caliper was the only caliper isolated, and only by its colour.
- **One colour, one group.**
  - Every red part of the Ducati (tank, fairings, mirrors, tail) is in "Red".
  - Every white armour piece of the Gundam is in one "Light gray" (398,578 px).
  - Every black part of the robot is in "Graphite".
  - The Torana's black hood shares a group with the chrome C-pillar trim, the front rim, the rear bumper and the exhaust (dE 2.3-8.4).
- **`absorb_lit` joining parts.**
  - The robot's shadowed left cuff, a "Cherry" cluster of its own (7,862 px, L 31), joined the red shoulders and feet (d_alb 3.1, d_pho 0.2).
  - The Gundam's darker torso and feet reds joined the shield's "Rose".
  - The van's gold mirror arm and cyclist decal joined the brass bumper (d_alb 3.6, d_pho 4.1).
  - The bicycle caliper's red face joined a rider's red item (d_alb 6.5, d_pho 6.7).
- **Half of it is the regions stage.**
  - The wheel split puts disc and rim into one region on both bikes.
  - One SAM region holds a whole shoe, a whole body side, or a whole Gundam head or shin.
  - Florence's lamp and ear boxes take their surroundings with them.

### 1.2 Junk groups (A)

**The code before this round.** The regions stage makes small regions, some on purpose and some by accident:
- the SLIC fill of pixels SAM leaves uncovered (`_fill_unlabeled`);
- small distinct proposals (source `'small'`, exempt from the speck merge);
- the object-side slivers of the matte cut (`cut_on_matte`);
- `recover_parts`, which prompts SAM on any chromatic pocket inside a neutral host, reflections included.

`cluster_colors` keeps any two regions apart when they are more than dE 10 apart, whatever their size. It compares global medians, so a 200 px shadow needs the same evidence as a 200,000 px panel.

Two rules could have re-joined such a piece, and both miss the colour-cast shadow:
- `absorb_lit` only moves chromatic candidates (normalised chroma >= 12, raw >= 18) into chromatic anchors (chroma >= 30), and never touches a neutral pair. Its material veto (duller than 0.6 of the anchor's chroma and more than 10 L darker) matches exactly what a coloured shadow looks like.
- `refine.lock_materials` applies the same veto to whole groups of the paint family and locks them as another material. The shadow becomes a locked row that a repaint of its paint cannot reach.

In addition, `refine.isolate_parts` locked every `'part'` region clustered into the paint family, with no test of surface against reflection. Complaint (A) therefore had two levers, `delta_e` and `absorb_lit`'s tolerances, and both are colour-only and size-blind.

**Measured.**
- 16 junk groups by eye (9,041 px, severity 25).
- Only 3 hits of the automatic test (4,537 px), one of them a paper sign that is not junk. Most junk sits on unlabelled object pixels, or is a reflection with another colour cast.
- 9 large lighting-split groups of one paint, and 41 part-level lighting splits.

By mechanism:

- **The owner's case:** the Torana's "Maroon" (222 px, severity 3), the shadow between the rear tyre and the flare. It is a SLIC superpixel dE 17.7 from the red. `absorb_lit` vetoed it as another material (chroma 0.5 of the anchor's, and darker), and `lock_materials` locked it.
- **Locked slivers.**
  - The BMW's black trim strip along the nose, tinted khaki by the yellow's bounce light ("Army green", 312 px, severity 2).
  - The Ducati's void through a sprocket lightening hole ("Chocolate", 194 px, hue within 25 deg of the red, severity 1).
  - The catalogue's Ducati "Crimson" (234 px) is listed as a fork-tube reflection at severity 3. The junk lane found it is actually the red Öhlins logo on the fork leg, a real decal. The entry is wrong, and it is still counted in every total below.
- **Just over the linkage threshold.**
  - The Fila's sole wedge in the shoe's own shadow, warmed by the ground (2,334 px, severity 2).
  - The van's underbody and sill shadow, warmed by the red floor (1,262 px in three matte-cut pieces, dE 11.8 to the body, severity 2).
  - The Fila's shadowed heel edge (dE 10.5).
- **Slivers and cavities.**
  - The Fila's lug gaps (400 px of ground shadow).
  - Two dark slivers and a pivot cavity on the Ducati. They were stamped as small distinct parts, which makes them exempt.
  - The robot's rim light along a leg (275 px, lighter than its host).
- **Reflections of other objects**, which the albedo carries: five groups on the Jaguar (2,769 px, severity 7). `recover_parts` accepted one of them as a part.
- **Large splits of one neutral paint**, which `absorb_lit` never touches:
  - the Jaguar's navy, in five groups; the body's best group covers 0.57 of it (sky reflection at L 90, shadow at L 40, blue-sky casts);
  - the Fila's white upper, in five groups; its tongue is split 52 / 47 between two;
  - the BMW's black rim, swingarm and engine cover, split by lightness (L 17 / 29 / 64);
  - the bicycle's grey frame highlights.

## 2. The reference set and the metrics

**The analyses.** Ten photos were analysed fresh at Balanced, in one process, through the live package. `ref/analyse.py` reproduces `recolor/pipeline.py` stage by stage: ingest, Careaga intrinsic, SAM 2.1 Balanced proposals, Florence-2 lettering and named parts, `hierarchy.build_regions`, `recover_parts`, the BiRefNet matte cut and `backdrop_decisions`, `grouping.group_regions`, and `refine.refine_groups` with the ViTMatte snap.

Each photo is cached under `ref/jobs/<name>/` with:
- the layers;
- the regions stage's partition and per-region sources;
- the matte, the SAM proposals and Florence's output;
- the groups before and after refinement;
- `trace.json`: the assignment of every input region after each rule (cluster, absorb_lit, absorb_washed, isolate_parts, absorb_highlights, split_decals, fill_decal_gaps), with each rule's log.

Peak VRAM was 17 GB in one process, all released afterwards.

**The sets** are under `ref/sets/<name>/`:
- `labels.npy` at the 1536 px work resolution: 0 other, 1..K parts, -1 ignore (excluded from every count);
- `parts.json`, with name, material, family, `must_be_separate`, provenance and ops;
- `object.npy`, the object silhouette: the job's BiRefNet matte unioned with every part.

`must_be_separate` is true when a customiser would realistically paint the part on its own: springs, calipers, discs, rims, tyres, levers and grips, mirrors, exhausts, seats, tank, fairing panels, grilles, badges, bumpers, decals, the sneaker's sole, panels and laces, every Gundam armour piece, and the robot's red, black and chrome pieces. It is false for lenses, windows, plates, small head details and the Torana's body panels (family `body`); only the Torana's hood and roof are true.

| photo | subject | parts / must-be-separate | masks |
|---|---|---|---|
| motorcycle_1 | Ducati 748, studio | 37 / 33 | round-3 hand-checked set (35), plus new SAM masks for the front disc (capped at L >= 45 to drop the carrier) and the right grip |
| motorcycle_2 | BMW S1000RR, yellow | 37 / 31 | round-3 set (36), plus the right grip |
| car_red_sports_1 | orange Holden Torana, showroom | 27 / 16 | round-3 set |
| car_red_sports_2 | red Porsche Boxster with hardtop | 14 / 10 | new. SAM returned rim + tyre as one mask, so the rims are ellipses fitted on the rim's dark step inside SAM's wheel, and tyres = wheel minus rim |
| car_classic_1 | Citroën H van, navy with gold lettering | 23 / 18 | new. Headlight lens and chrome housing: SAM, a morphological opening, and a hand rectangle cutting the mount stalk. Front rim: a hand ellipse (rim prompts returned the whole wheel). The four gold decals: CIELAB selections (box prompts returned the whole panel). Rear wheel dropped (deep shadow) |
| car_classic_2 | Jaguar XK150, navy, front view | 17 / 10 | new. Bumper: SAM, a hand polygon over its dark reflection band, and a hole fill. Lamp pods: SAM minus their lenses |
| bicycle_1 | Cube road bike, panning shot | 9 / 8 | new. Rear wheel ring cut to a hand ellipse annulus, and not must-be-separate (hidden by the other bike). Rear derailleur dropped (blur) |
| sneakers_2 | white Fila Disruptor | 12 / 12 | new; it replaces the lime clogs, which have no personalisable parts. Logos: CIELAB selections. Laces and eyestay: SAM with 8-12 clicks. Upper panels split at the stitched seams |
| robot_toy_2 | tin Robby | 20 / 18 | new. Left ear: a hand ellipse band (SAM returned half the ring) |
| gundam_rx78_rg | RG RX-78-2, CG product render | 43 / 38 | round-3 set |
| **total** | | **239 / 194** | |

**How the sets were built.** `ref/specs/<name>.json` are reproducible op lists run by `refbuild.py`. The ops are:
- the SAM 2.1 hiera-large image predictor on zoomed crops (up to 4x, 1024 px input), with positive and negative clicks and boxes;
- the mask logits resized back to full size with hysteresis thresholding;
- an optional GrabCut band snap;
- hand ops: polygons, rectangles, CIELAB flood fill and selection, fitted or hand-drawn ellipses, and seam splits.

The round-3 sets (provenance `gt`) were built the same way and reviewed on 3-5x crops; their ignore pixels are carried over. The five new sets were checked by eye on per-part debug thumbnails, on 3x tile sheets along every part edge, and on 4-5x crops of every hand-corrected part.

The integration later added two cars with painted calipers, built the same way: `car_alpine_1` and `car_corvette_1`. They add 27 must-be-separate parts, making the twelve-photo set 221.

**Metrics** (`ref/partkit.py`; thresholds are module constants). For each must-be-separate part, the *best group* is the group holding most of its pixels. *Coverage* is that group's share of the part; *purity* is the part's share of the group.
- **part_isolation** (the headline): the share of parts with purity >= 0.8 and coverage >= 0.7, meaning the part can be painted alone by picking one group.
- **region_isolation:** the same test on the label map. This is the ceiling any grouping of those regions can reach.
- **merged:** the best group is also another must-be-separate part's best group, or holds more than 20 % of its pixels outside every part.
- **fragmented:** coverage < 0.7.
- **junk_groups** (automatic): groups under 0.4 % of the object, at least 90 % inside one part, whose lightness-normalised (a, b) is within 12 of that part's main group.
- **lighting_splits:** the same test without the size cap.
- **tiny groups:** under 0.4 % of the object.
- **object_slivers:** tiny object groups whose ring neighbour has the same body colour. These need no reference part.

The by-eye severity scale (`catalogue.py`, a single pass on 3x sheets and 4x crops):
- 0: not junk;
- 1: a cosmetic sliver (under 0.1 % of the object, no visible seam after repainting the parent);
- 2: a visible extra swatch row, or a seam on the part;
- 3: the parent's repaint leaves a visibly different patch, or the group is locked or background.

The catalogue holds 16 junk groups and 17 real small groups. A later grouping is matched to it by pixels: a catalogued group counts as removed when 90 % of its pixels sit in larger groups.

The lanes added:
- **kind-level isolation:** the best group covers >= 70 % of the part and is >= 80 % parts of that kind, counted over the 133 must-be-separate parts whose kind is in the photo's vocabulary;
- **stamped-mask precision:** a mask at least half on a reference part of its own kind is true, at least half on another part is wrong, and anything else is unlabelled;
- the round-3 `evalkit` region metrics, on `ours@quick` (the six round-3 reference photos) and `paco@quick` (21 PACO images);
- renders through the live `Renderer` (paint family navy `#123f9e`, targets red `#e63946`), scored as *collateral*: object pixels outside the targets that turn red.

**Baseline** (`ref/baseline.json`, sheets `ref/sheets/<name>.jpg`; every lane's control reproduces these group maps):

| photo | must-be-separate | isolated | region isolation | purity / coverage | merged | fragmented | junk by eye (px, severity) | groups (background, locked) |
|---|---|---|---|---|---|---|---|---|
| motorcycle_1 | 33 | 1 (tank decal) | 0.61 | 0.20 / 0.90 | 32 | 5 | 4 (1,189, 6) | 25 (7, 2) |
| motorcycle_2 | 31 | 2 (front caliper, RR decal) | 0.52 | 0.16 / 0.88 | 29 | 6 | 1 (312, 2) | 12 (1, 4) |
| car_red_sports_1 | 16 | 0 | 0.62 | 0.25 / 0.89 | 13 | 3 | 1 (222, 3) | 43 (23, 2) |
| car_red_sports_2 | 10 | 0 | 0.30 | 0.21 / 0.86 | 10 | 3 | 0 | 20 (13, 0) |
| car_classic_1 | 18 | 1 (body) | 0.39 | 0.15 / 0.88 | 18 | 3 | 1 (1,262, 2) | 16 (9, 0) |
| car_classic_2 | 10 | 1 (Jaguar badge) | 0.30 | 0.24 / 0.82 | 9 | 3 | 5 (2,769, 7) | 31 (15, 0) |
| bicycle_1 | 8 | 0 | 0.25 | 0.22 / 0.77 | 8 | 3 | 0 | 16 (2, 0) |
| sneakers_2 | 12 | 0 | 0.08 | 0.10 / 0.92 | 12 | 1 | 3 (3,012, 4) | 22 (11, 0) |
| robot_toy_2 | 18 | 0 | 0.67 | 0.16 / 0.97 | 17 | 0 | 1 (275, 1) | 29 (17, 0) |
| gundam_rx78_rg | 38 | 1 (blue chest) | 0.55 | 0.12 / 0.98 | 37 | 0 | 0 | 16 (7, 0) |
| **total** | **194** | **6 (0.031)** | **0.490** | **0.181 / 0.887** | **185** | **27** | **16 (9,041, 25)** | **230 (105, 8)** |

Also in the baseline: 3 automatic junk groups (4,537 px, severity 7), 79 tiny groups (30 on the object) and 7 object slivers (8,041 px).

**Caveats**, which every number inherits:

1. **part_isolation is strict by construction.** A colour-perfect grouping can never isolate a red mirror from a red tank. That is the point, so the headline moves only with part-aware grouping. Per-kind groups (both mirrors in "Mirrors") also count as merged, which is why the kind-level and per-instance figures are reported too.
2. **region_isolation is the ceiling for the current regions.** 99 of the 194 parts never get a region, so half of the problem lies in the regions stage.
3. **must_be_separate is a judgement.** The Torana's body panels are not flagged. The Boxster's and the Jaguar's bodies and hoods are, because they were built as single parts. `parts.json` carries a `family` field that the metrics do not use.
4. **The masks favour SAM-based candidates.** They are SAM-drawn wherever no hand op overrode them.
   - The new cars' rims include the spoke voids, and their tyre boundary is a fitted ellipse.
   - The Boxster's calipers, the bike's rear derailleur and the van's rear wheel are not resolvable at 1536 px, so they are "other".
5. **The automatic junk test is deliberately narrow** (3 hits, 1 false). By-eye severity is the reference; object slivers and lighting splits are the automatic proxies.
6. **The by-eye verdicts are a single pass.** Severity 1 against 2 on a sliver is a judgement call, and one entry (the Öhlins logo) is wrong.
7. **One fresh analysis per photo**, from the 2026-09-29 tree. Group ids change on re-analysis. The Gundam is a CG product render, not a photo of a built kit.
8. **No held-out set.** Every threshold of every lane and of the integration was tuned on these ten photos (the integration also on the Alpine and the Corvette).

## 3. Experiments

Two surveys came first; section 7 lists every candidate with its licence and verdict.

**The parts survey** ranked the detectors:
1. SAM 3 concept prompts: masks, a presence head and exemplars, but the weights are gated.
2. Florence-2's per-phrase open-vocabulary detection: no new weights needed.
3. LLMDet / MM-Grounding-DINO and OWLv2.

It also set out the plumbing any detector needs:
- keep the kind at stamping;
- carry it on `Region` and in `regions.json`;
- cluster part regions apart from colour;
- keep part groups out of `absorb_lit` and `absorb_washed`;
- name each group after its kind.

**The junk survey** found three signals missing from the grouping: a boundary classifier, a size or significance rule, and a part prior. It collected twelve sanity rules from the literature. It ranked three proposals first: a size-aware linkage threshold, quasi-invariant boundary classes, and paired-region ratios.

Three lanes then ran on the reference set, each in `scratch/groupexp/exp/<key>/`, and a verifier re-ran each one.

| config | isolated / 194 | kind-level / 133 | merged | fragmented | by-eye junk left / 16 (severity / 25) | real small lost / 17 | auto junk | tiny (on object) | object slivers | groups (background, locked) | cost | verifier; verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** (reference analysis) | 6 (0.031) | 2 | 185 | 27 | 16 (25) | – | 3 | 79 (30) | 7 | 230 (105, 8) | – | – |
| **parts** `A_owl` (OWLv2 only, score >= 0.3, SAM >= 0.85, one group per kind) | 20 (0.103) | 24 | 165 | 26 | 15 (24) | 0 | 3 | 85 (35) | 11 (4 are part groups) | 258 (105, 8) | part step 1.23 s per image; 3.6 GB | minor; improves |
| parts `O_inst` (one group per instance) | 29 (0.149) | 25 | 156 | 26 | 15 (24) | 0 | 3 | 86 (36) | 11 | 266 (105, 8) | same | – |
| **junk** `all_0.004` | 6 (0.031) | – | 185 | 27 | 11, 1 partial (16) | 0 | 2 | 74 (25) | 5 | 225 (105, 6) | 0.32 s per image, CPU only | minor; improves (modest) |
| junk + backdrop crumbs | 6 | – | 185 | 27 | 11 (16) | 0 | 2 | 40 (25) | 5 | 191 (71, 6) | same | – |
| **stack** (parts per kind + part-aware pruning) | 20 (0.103) | 24 | 165 | 26 | 10, 1 partial (15) | 0 | 2 | 79 (29) | 8 (4 are part groups) | 252 (105, 6) | about 2.7 s per image fresh; 3.6 GB | minor; improves |
| stack, per instance | 29 (0.149) | 25 | 156 | 26 | 10 (15) | 0 | 2 | 80 (30) | 8 | 260 (105, 6) | – | – |
| stack + backdrop crumbs | 20 | 24 | 165 | 26 | 10 (15) | 0 | 2 | 45 (29) | 8 | 218 (71, 6) | – | – |
| stack without the part guard | 19 (0.098) | 23 | 166 | 26 | 10 (15) | 0 | 2 | 75 (25) | 5 | 248 (105, 6) | – | – |

Region isolation is 0.490 in every row, because the lanes start from the reference partition. The baseline's kind-level figure is the parts lane's control. The stack lane's scorer prints 0 there, because its vocabulary comes from a run's own caption, and a run without detected parts has no caption.

### 3.1 `parts`: detected parts as groups of their own

**What was tried.**
- **Vocabulary.** A Florence-2 `<CAPTION>` picks the object class: motorcycle, car, bicycle, sneaker, figure or generic. All ten photos were classified correctly. The class selects a vocabulary of part kinds, each with 1-2 bare-noun prompts, a display name, a size range per instance (as a share of the object) and an instance cap. `panel` kinds such as tank, fairing, hood and roof are off by default.
- **Detector probe (t01)** on all ten photos, each detector run on the work image plus four 0.6 corner tiles:
  - Florence-2 `<OPEN_VOCABULARY_DETECTION>` per phrase, with a confidence built from its location-token probabilities;
  - OWLv2-large-ensemble;
  - Grounding DINO base.
- **Masks.** SAM 2.1 box prompts through the live `SamMasker.prompt_boxes`, followed by gates.
- **Stamping** into the regions-stage partition after `cut_on_matte`, in three generations:
  - v1: plain pixel stamping;
  - v2/v3: adopt an existing region at IoU >= 0.6; keep regions the mask only nicks and distinct sub-parts inside it; always cut a host more than twice the mask; join remnant rings to the part only when they are the part's colour.
- **Part-aware grouping.** Part regions are held out of `cluster_colors` and the backdrop split. There is one group per kind (or per instance), named after the kind, held fixed through `absorb_lit` and every refine step. During the refine steps the groups are locked, `enforce()` runs after each step, and part pixels are protected from the snap; the groups are unlocked at the end.
- **A sweep of 15 configurations:** detector thresholds, SAM 0.70 / 0.92, loose gates, each detector alone, the panel tier, region-snapped stamping, and per kind against per instance.
- Engine renders, and `evalkit` on the regions stage with and without stamping.

**What happened.**
- **Detectors.**
  - Florence-2's open-vocabulary detection is unusable for parts. Most phrases return a tile-sized box, and its confidences do not separate right answers from wrong ones.
  - OWLv2 finds small parts: the Ducati's spring at 0.48, the grip at 0.40, the sprocket at 0.45.
  - Grounding DINO is best on wheels and seats, and misses small parts.
  - Neither scores any brake caliper above 0.2 on the three bikes.
- **Gates and stamping.**
  - v1 raised isolation from 6 to 17, but put 9 of the Ducati's 20 masks on the wrong part. Region isolation fell from 0.490 to 0.464: the front tyre and both front calipers lost their regions to the stamped wheel.
  - Grounding DINO's wheel boxes are what cost region isolation.
  - The detector threshold is the sensitive knob: at 0.2 or lower, the number of wrongly placed masks rises from 5 to 33-61.
- **The chosen `A_owl`** (OWLv2 only, score >= 0.3, SAM >= 0.85, one group per kind):
  - Isolated parts went from 6 to 20 (0.103), or 29 (0.149) with one group per instance.
  - Kind-level isolation went from 2 to 24 of 133. Merged parts went from 185 to 165.
  - Region isolation is unchanged at 0.490. Purity went from 0.181 to 0.282.
  - No new junk. The untagged tiny groups stay at 79 and the untagged slivers at 7. The 6 new tiny groups are all part groups (Grip, Footpeg, Door handle, Badge, Fuel cap, Logo), and 4 of them are also the 4 new slivers.
  - 43 masks were stamped: 32 on a reference part of their kind, 5 on another part, 6 on unlabelled pixels.
- **evalkit.** On ours, every key is within 0.001 of the control, except best IoU (+0.004) and R@.75 (+0.011). On PACO, best IoU went from 0.337 to 0.348, R@.5 from 0.327 to 0.343, and achievable recall from 0.388 to 0.408.
- **By photo.** Ducati 1 -> 6 (shock spring, grip, seat, rear sprocket, front tyre); BMW 2 -> 5; Torana 0 -> 3; van 1 -> 3; Jaguar 1 -> 2.
- **Renders** (collateral, CURRENT -> NEW):
  - Ducati spring + grip: 18,806 -> 245 px. Only the spring turns red; the gold shock body and frame stay gold.
  - Torana rims: 12,824 -> 45 px. The black hood no longer turns red.
- **Cost.** 1.23 s per 1536 px image for the part step (0.36 s at 640 px), and 3.6 GB reserved for the whole regions stage plus parts in one process.

**Verifier corrections applied.** Every number reproduces:
- `run_metrics.py` reproduces the lane's numbers.
- A fresh in-process re-run of `A_owl` gives bit-identical group maps on all ten photos.
- A fresh OWLv2 pass on the Ducati matched the cache: 38 of 38 boxes at >= 0.3, with a best caliper score of 0.186.

Four corrections, applied above:
1. **The BMW rims render** (133,731 -> 617 px by a redness test) is a precision gain with a recall loss. By group map, the "Wheel rim" group covers 0 % of the front rim and 74 % of the rear (CURRENT's groups covered 69 % and 96 %), and it spills about 5.5k px onto the rear tyre.
2. **Coverage beside the spill.** The Torana's "Wheel rims" covers 0.468 of the rear rim (CURRENT: 0.532); its polished lip stays silver. The grip's coverage goes from 1.000 to 0.930. The spring stays at 0.999 and the Torana front rim at 1.000.
3. **Photos that gain nothing.** Strictly, five photos gain nothing, not four: the Boxster goes 0 -> 0 strict (0 -> 3 kind-level).
4. **Painted parts leave the paint family.** A painted part detected as a kind leaves the paint family. The robot's feet go from 1.000 in the paint family to 0.001, so a repaint of the body leaves them unchanged.

**Verdict.** Improves, with a narrow reach; minor corrections.
- The mechanism is safe by construction: part regions stay outside every colour step, each kind is one unlocked group, and part pixels are kept out of the snap. It adds no junk and costs nothing in segmentation.
- Its reach is limited by the detector. Not solved: brake calipers (0 of 3 on the bikes), sneaker panels and laces, Gundam armour, robot arms, body panels.

**What shipped from it:** the whole mechanism with OWLv2 only, at score >= 0.3 and SAM >= 0.85; per-kind groups with Split by instance; and the `Region` part fields (section 4). Grounding DINO and the panel tier were dropped.

### 3.2 `junk`: pruning tiny lighting-variant groups

**What was tried.** One post-step after `refine.refine_groups`. It only moves regions between groups, so the label map, the snap and the decal islands stay valid.
- **T0** reproduced the baseline exactly.
- **T1** built an evidence table for every group under 2 % of the object. Each group was compared with every touching group at the seam (its pixels against the neighbour's in a 5 px ring), in the photo, the shading layer and the albedo.
- **Candidates:** non-background groups under 0.4 % of the object, at least 60 % on the matte, smallest first, each region judged on its own.
- **Three tests:**
  - **Shadow.** All of these must hold:
    - the neighbour owns >= 30 % of the ring;
    - the photo is darker (ratio <= 0.85);
    - the shading layer carries >= 30 % of the log-luminance step;
    - the chromaticity moved the way light moves it: no more chromatic (+3) with the hue kept within 35 deg, or the photo's chromatic log-shift equals the shading's within 0.08;
    - against a chromatic neighbour (C >= 20), the region keeps >= 0.6 of its chroma.
  - **Gradient.** No photo edge on the shared boundary (median |grad log Y| and |grad log chromaticity| <= 0.12), albedo cast <= 8, photo dE <= 6.
  - **Significance.** Under 0.4 % of the object, not locked, and its closest-coloured neighbour has the same body colour (lightness-normalised (a, b) within 12) with photo dE <= 6.
- **Exempt:** lettering, named parts, the wheel split, recovered parts, and groups that are at least half decal islands.
- **T2-T5** fixed what the first passes got wrong. **T6** swept the size limit from 0.1 % to 5 % of the object. **T7** moved one knob at a time (13 configurations).
- Also tried: a residual-layer reflection test; engine renders of every move; an optional `backdrop_crumbs` for tiny background groups.

**What happened.**
- **What separates junk.** Seam statistics separate junk from its host; global medians do not. The Fila's shadowed sole wedge is 0.4 from its neighbour at the seam, but over dE 10 from its group's median.
  - The shading layer carries 37-65 % of the log-luminance step on real shadows, and only 5-22 % on dark real parts.
  - Photo edge strength does not separate junk from parts.
- **Fixes along the way:**
  - The first pass made 9 merges, 3 of them wrong. Two Gundam and one robot backdrop pieces that the matte had not flagged went into the red paint, and the robot's dome mechanism went by the gradient test. This led to the matte and photo-dE conditions.
  - Group-level moves sent the van's three-piece shadow group whole into the gold bumper. This led to judging each region on its own.
  - Region-level colour moves nibbled real groups: a door-handle recess, a rider's glove taken for a highlight, a Gundam gap 50x darker than the red. This led to three changes: the colour rules dissolve whole groups only, the highlight branch is off, and a chroma floor was added.
- **The render overruled the metric once.** Chroma retention 0.4 scores better (10 junk groups left). But it joins the van's floor, seen through the bumper frame (retention 0.54), to the bumper, which puts a visible navy wedge in the frame under a bumper repaint. At 0.6 the floor stays out, while accepted shadows retain 0.70-0.73.
- **Size.** 12 / 12 / 11 / 11 / 11 / 11 junk groups left at 0.1 / 0.2 / 0.4 / 0.8 / 2 / 5 %.
  - At 2 %, a real 21.8k px dark group joins the Jaguar's body.
  - At 5 %, pieces of the RX-78's rifle join the white armour.
  - Above part scale, the shadow test cannot tell a dark neutral material from a shadow on a neutral paint.
- **The residual layer does not separate reflections.** Their residual share is -0.1 to 0.4, against 0.3 for real badges; the albedo carries the reflections.
- **The chosen `all_0.004`:**
  - By-eye junk went from 16 to 11 (5 removed, 1 partial); severity from 25 to 16.
  - Automatic junk went 3 -> 2, slivers 7 -> 5, tiny groups 79 -> 74, groups 230 -> 225, locked 8 -> 6.
  - No real small part lost (0 of 17). No must-be-separate part changed isolation (front_sole purity 0.149 -> 0.155).
  - Removed: the Torana's locked flare shadow (the owner's case), the BMW's locked trim edge, and the Fila's sole wedge, lug gaps and heel edge. Partial: the van's sill band (the floor through the bumper frame correctly stays).
  - With `backdrop_crumbs`: 191 groups (background 105 -> 71) and 40 tiny groups, with no change on the object.
- **Renders.** The colour cast between the moved pixels and the paint band around them:

| crop | original | current | new |
|---|---|---|---|
| Torana flare | 19.9 | 39.1 | 22.1 |
| BMW nose edge | 12.6 | 78.8 | 25.2 |
| van sill | 2.3 | 30.9 | 10.5 |
| Fila sole wedge, first region | 0.9 | 21.1 | 4.5 |
| Fila sole wedge, second region | 2.0 | 22.5 | 4.2 |
| Fila heel edge, first region | 1.5 | 17.3 | 9.0 |
| Fila heel edge, second region | 2.9 | 6.1 | 5.5 |
| Fila lug gaps, first region | 9.7 | 30.3 | 24.8 |
| Fila lug gaps, second region | 9.8 | 25.0 | 20.1 |

  Outside a 7 px band around the moves, CURRENT and NEW differ by 14 px on the BMW and 0 px elsewhere.
- **Cost:** 0.32 s per photo, on the CPU.

**Verifier corrections applied.** Every number reproduces:
- A deep diff of `run_metrics.py` output against `result.json` finds no difference.
- `prune_junk` re-run from scratch gives bit-identical group maps.
- The label map is unchanged, there are no -1 pixels, and every region is in exactly one group.

Two corrections, applied above:
1. **All nine render crops, not the five best.** The lug gaps stay far from the original after the move, and one heel piece barely changes.
2. **The Fila's lug gaps went into the wrong group.** They are shadowed ground seen through the tread notches, and they went into the white shoe's group at a photo ratio of 0.01-0.017. The shadow test has no darkness floor on a neutral host (the chroma floor only protects chromatic hosts). This is the same failure the lane itself describes above 0.4 %.

**Verdict.** Improves, modestly: 5 of 16 junk groups and 10 regions (about 4 kpx) across ten photos, with no part loss. Two kinds of junk need other signals: reflections of other objects, which live in the albedo, and cavity shadows stamped as `'small'` islands, which are exempt on purpose. The lane also found that the catalogue's Ducati "Crimson" is the Öhlins logo; without it, the catalogue totals would read 22 -> 13.

**What shipped from it:** the three tests with these thresholds, run at the end of `refine_groups` and `regroup_refined`, with `backdrop_crumbs` on for product shots. The integration added the part rules, a part-rim test and a crumbs rule (section 4.3).

### 3.3 `stack`: both lanes together, and the Groups panel model

**What was tried.** The parts lane's `A_owl`, followed by the junk lane's three tests, re-hosted with two stacking rules:
- **Rule 1:** every group holding a part region is exempt as a candidate.
- **Rule 2:** a pruned candidate at least 50 % under a part's SAM mask (dilated 2 px) joins that part's group, and part groups host nothing else.

Variants: without the guard; with part groups open as hosts for every candidate; per instance; with backdrop crumbs.

A Groups panel model (`panel.py`) with sections, names, Minor rows with parents, a render view with an opt-in "Paint with <parent>", and `split_instances` checked against the per-instance run. Also a fresh `evalkit` run and downstream renders on four photos.

**What happened.**
- **The lanes compose without interference.**
  - Isolated parts: 20 (29 per instance). Merged: 165.
  - By-eye junk: 10 left (severity 15). No real small part lost (0 of 17). Automatic junk: 2.
  - Groups: 252, or 218 with crumbs, which is below the baseline's 230.
  - That is 5 fewer junk groups than parts alone and 14 more isolated parts than junk alone. It also removes one junk group more than junk alone: a Jaguar reflection sliver, re-clustered away by the stamping.
- **Rule 1 is load-bearing.** Without it, the tests dissolve 4 of the 35 detected parts and isolation drops to 19:
  - the Ducati's grip, as a shadow of the bar;
  - the BMW's footpeg, the Jaguar's "Fuel cap" and the Fila's logo, as insignificant.
- **Rule 2 never fires.** Opening every part group as a host gives the identical result. Only 3 of the 39 leftover tiny object groups lie under a part mask (a sprocket hole, the Öhlins decal, a badge's enamel), and none of them is a shadow.
- **Prune moves.** Of the 11 moves, two are not covered by the catalogue; both were checked on 3x crops and are correct: the van bumper's 152 px shadowed top edge, and a 343 px Gundam white-armour piece.
- **Split.** `split_instances` on all 8 multi-instance rows reproduces the per-instance partition on all ten photos.
- **"Paint with <parent>" became opt-in.** Of the 5 Minor rows it would have made follow their parent, 2 must not: the van's floor through the bumper frame, and the paper sign behind the Torana's glass.
- **Minor rows.** Minor held 34 rows plus 2 part-detail rows: 10 junk, 12 real small parts and 14 uncatalogued.
- **Clutter.** Rows visible by default: 252 flat, 151 with sections. For today's grouping the same two numbers are 230 and 112.
- **evalkit.** All keys are within 0.002 of the live regions stage. On ours, best IoU is +0.004 and R@.75 +0.011; on PACO, best IoU is +0.011, R@.5 +0.016 and achievable recall +0.020.
- **Renders** (collateral, CURRENT -> NEW):
  - Torana rims: 28,911 -> 876.
  - Gundam helmet + shield: 445,596 -> 64,387. The "Head" row is the whole head.
  - BMW calipers: 46,553 -> 46,549. No caliper was detected.
- **Cost.** About 2.7 s per image fresh (1.22 s parts step, 1.17 s groups stage, 0.29 s pruning), against 1.24 s for the live groups stage. Peak VRAM 3.6 GB (evalkit), 2.9 GB for the stack runs.

**Verifier corrections applied.** All 1,682 leaf values of `run_metrics.py` match. A fresh in-process re-run of `stack` and `stack_noguard` reproduces every group and label map, and confirms exactly the 4 parts pruned without the guard.

Corrections:
1. **The Ducati's spring + caliper collateral** (40,836 -> 16,049) hides a coverage regression.
   - The caliper's holder group covers 0.70 of it (it was 1.00), and 30 % of the caliper sits inside the "Wheel rim" part row.
   - The painted target share falls from 0.999 to 0.889.
   - The holder group ("Sand", 7 % caliper) also reddens a rear-wheel spoke fragment and the headstock.
2. **The Torana's rear rim** is split 53 / 47 between "Khaki 2" and "Wheel rims". The 876 px figure needs both groups painted; "Wheel rims" alone leaves half the rear rim unpainted.
3. **The Minor tally** is 34 + 2 = 36.
4. **evalkit** is within 0.002, not 0.001 (bdist moved by +0.0018 px).

**Verdict.** Improves; minor corrections. Brake calipers are not fixed: no detector finds one on any bike, and the Ducati caliper's footprint is scattered rather than usable.

**What shipped from it:**
- rules 1 and 2;
- the panel sections: Parts, Colours, Minor (collapsed), Background (last);
- Split by instance, named by position;
- backdrop crumbs on;
- the `Region` kind and instance fields.

Not shipped: the "Paint with <parent>" toggle. Minor rows got "Merge into <parent>" instead.

## 4. What shipped and how it behaves

The integration put the stack into `recolor/` and then fixed what its checks found. Beyond the lanes it added:
- a second look at every wheel, for brake calipers and discs;
- the second wheel of each bike;
- a mirror-stalk fold;
- a check that the matte kept one subject;
- a part-rim rule and a crumbs rule in the pruning;
- user votes that survive a regroup;
- a sheen merge;
- a lighting-variant rule for the Minor section;
- front/rear instance names.

Where things live: OWLv2 is in `partdetect.py`, the subject check in `subject.py`, the pruning in `junk.py`, and the part logic in `smallparts.py`, `wheels.py`, `grouping.py` and `refine.py`. The panel is in `web/js/panels/groups.js`.

### 4.1 Detected parts, in the regions stage

`smallparts.find_kind_parts` and `stamp_parts` are run by `pipeline._detect_parts` at Balanced and Max, after the matte cut and before the backdrop decisions. Fast skips this step. Without OWLv2 or without the caption, the stage runs as before and logs why.

- **Vocabulary.** Florence-2's `<CAPTION>` picks the object class: the first class word in the caption (motorcycle, car, bicycle, sneaker, figure), else generic. The class selects a list of part kinds (`smallparts.VOCAB`). Each kind has 1-2 bare-noun prompts, a label and plural, a size range per instance (as a share of the object) and an instance cap.
  - **Motorcycle:** shock spring, brake caliper, brake disc, wheel rim, tyre, mirror, exhaust, seat, grip, lever, sprocket, fork, footpeg.
  - **Car:** wheel rim, tyre, brake caliper, grille, mirror, badge, exhaust tip, door handle, bumper, spoiler, fog lamp, vent, fuel cap.
  - **Bicycle:** rim, tyre, brake caliper, brake disc, saddle, handlebar, crankset, pedal, bottle, fork.
  - **Sneaker:** sole, laces, logo, tongue, heel counter, toe cap.
  - **Figure:** head, antenna, shoulder armour, shield, weapon, beam saber, hand, foot, ear.
  - **Generic:** logo, handle, button, strap.
  - **Panel kinds** (tank, fairing, fender, hood, roof, frame, arm, leg) are asked for so that they claim their own boxes, but they are never kept. They would split the one paint a user repaints in one click, and they cost region isolation when measured.
- **Detector.** OWLv2 large-ensemble (`partdetect.py`, pinned snapshot `95e2693`, fp16, local Hugging Face cache only) on the work image and its four 0.6 corner tiles, with every phrase of the class in one pass per crop.
- **Gates** (`PartGates`):
  - OWLv2 score >= 0.3.
  - Per-kind box NMS: IoU 0.6, or 80 % inside another box.
  - A box at most 4x the kind's largest mask.
  - At most 12 SAM prompts per kind, as SAM 2.1 box prompts on crops (the box + 8 % + 8 px), in chunks of 16 against a 6 s budget.
  - A mask is kept when all of these hold: it is not clipped by its crop; SAM's score is >= 0.85; its box has IoU >= 0.45 with the detection and fills >= 0.1 of it; >= 80 % of it is on the matte; it fits the kind's size range.
  - Duplicates within a kind (IoU > 0.5, or 80 % inside) and across kinds (IoU > 0.6) keep the best mask. Then the instance cap applies.
- **Wheels.** A wheel box goes through `wheels.split_wheel`, which splits it into a tyre and a rim along the rim's lip ellipse.
  - A wheel adds nothing if it is under 1.5 % of the object, if its mask fills less than 30 % of its box, or if it does not split.
  - The outer ellipse can also pass on angular support: an inlier in 60 % of its 5-degree sectors. This is for the Ducati's rear wheel, notched deep by the swingarm, chain guard and fender.
  - A lip fallback lets every edge of the outer band compete, for the BMW's black front rim in its black tyre.
  - The rim keeps only its pixels on the matte.
- **The wheel second look** (`find_calipers`, for classes with a caliper kind). No detector scores a caliper at the photo's scale (the best on the Ducati is 0.186), so every accepted wheel is looked at again.
  - **Candidates**, inside the wheel's convex hull: OWLv2's caliper boxes on a crop of the wheel, and compact regions of the partition.
  - **Verified** candidate: OWLv2 on a square crop 2.5x its size names a box agreeing with it as a caliper at >= 0.22.
  - **Painted** candidate: it has a clear colour (chroma >= 20) that covers at most 5 % of the rest of the wheel and of a band around it, and does not continue outside the wheel.
  - One caliper per wheel, plus its pieces within dE 12.
- **The disc** (`find_discs`, motorcycle and bicycle). The same pass looks for the disc inside a split wheel.
  - A disc box centred on the hub (within 0.3 of the radius) and 0.4-0.95 of the wheel's size is prompted to SAM.
  - SAM's answer is kept when it lies 85 % inside the rim and is at most 0.7 of the rim, with at most a quarter of it in the rim's outer 15 % (the lip).
  - Holes and specks under 25 px are closed.
- **Stalks** (`attach_parts`). A smaller part touching a detected mirror, at most half its size and inside its box grown by its size, is the mirror's stalk and is folded into the mirror. OWLv2 had called the Alpine's mirror stalk a "rear spoiler". A grip beside a bar-end mirror is never folded.
- **Stamping** (`stamp_parts`). Larger parts go first, smaller ones over them.
  - A region with IoU >= 0.6 with the mask is adopted whole. A tyre or a rim adopts only regions lying 90 % inside it: the BMW's front tyre region had reached over the fork stanchion, and "Tyres" painted the fork (integration).
  - Otherwise the mask is stamped pixel by pixel, except over:
    - lettering;
    - other parts;
    - regions it only nicks (less than half inside);
    - distinct sub-parts inside it (>= 85 % inside, <= 35 % of the mask, >= dE 15: a caliper on a rim).
  - A host more than twice the mask is always cut. The remnant ring of a cut region joins the part when it is the part's colour (dE 15).
  - A mask built from regions adopts every region lying 85 % inside it within dE 10. This is for the Corvette's caliper, seen above and below a spoke.
  - Every instance is one region of source `'kind'` carrying `part_kind`, `part_label`, `part_plural`, `part_instance` and `part_score`. It is always object. The SAM masks stay in the regroup seed for the pruning.
- **One subject** (`subject.other_objects` + `cut_off`, `pipeline._subject_matte`, for car, motorcycle and bicycle captions). The matte takes the salient object and whatever touches it. The red coupe parked behind the Torana had its door inside the matte and was painted with the Torana.
  - SAM on the matte's box gives the subject's silhouette, when it scores >= 0.9, lies 90 % on the matte and covers 85 % of it.
  - SAM, point-prompted inside each leftover piece of the matte (>= 2 % of the subject and >= 1,500 px), says whether the piece is another object: its answer covers 80 % of the piece and lies at most 5 % inside the subject.
  - Such pieces go to the backdrop, together with every region lying half in them.
  - Detected parts are never touched, and a pair of sneakers is not examined.
  - Cost: 0.04-0.14 s (integration).

### 4.2 Part groups, in the groups stage

- **One group per kind.** A region with a part kind never enters the colour clustering. Each kind is one group whatever its colour and however small. It sits outside the `max_groups` cap, is never background, and is unlocked and paintable.
  - A group is a *part group* when more than half of its area is one kind's part regions (`ColorGroup.part`, `part_label`, `part_plural`, `part_instances`). This is recomputed on every rebuild, so the flag follows merge, split, move and regroup.
  - Its automatic name is the kind's label or plural ("Shock spring", "Wheel rims"). A custom name is kept.
- **Kept out of every colour rule.** A part group:
  - is never a candidate or an anchor of `absorb_lit`;
  - is never the paint (`main_paint`, `paint_family`);
  - never absorbs a region through, or loses one to, `absorb_washed` or `absorb_highlights`;
  - is never locked as another material;
  - never takes in a decal or a carried-over region by colour;
  - never has its pixels moved by the ViTMatte snap.

  `enforce_parts` restores the invariant (every part region in its kind's group, and nothing else there) after the snap and after a regroup's carry-over.
- **Split by instance** (`split_instances`; `POST /api/jobs/{id}/groups/split` with `mode: "instances"`). This makes one group per instance with no colour clustering, so two calipers of the same colour split cleanly.
  - Instances are named by position: "(left)" / "(right)", "(upper)" / "(lower)", and numbered beyond two.
  - On a vehicle, rims, tyres, calipers, discs, door handles, footpegs, seats and exhausts are named "(front)" / "(rear)" when the parts give the front away. `vehicle_front` uses the two wheels' midpoint, with the grille, fork, grips and mirrors voting for the front and the sprocket, spoiler, exhaust and seat for the rear.
  - Every instance keeps the part's albedo as its reference colour (`ref_lab`), so the split alone changes no pixel.
  - Merging instances with automatic names gives the kind's name back.

### 4.3 Junk pruning, and the sheen merge

`junk.prune_junk` is the last step of `refine.refine_groups` and of `refine.regroup_refined`, with `junk.JunkParams` defaults. It only reassigns regions: the label map, the region ids and the snap stay valid, moved regions leave the decal islands, and the protect mask is recomputed.

- **Candidates:** the regions of every non-background group under 0.4 % of the object (matte > 0.5), at least 60 % on the matte. Groups go smallest first, and each move is visible to the next. Every comparison is the region's pixels against the neighbour's in a 5 px ring: at the seam, not against a median.
- **Tests:**
  - **0, part rim.** A region lying 90 % within 4 px of a detected part's SAM mask is the part's shadowed edge and joins the part, when all of these hold:
    - the part owns at least half of the region's object ring;
    - the region is no lighter and no more chromatic than the part (+5);
    - for a coloured part, it is in the part's hue (within 30 deg).

    The reason: SAM's box answer stops a few px short of the outline. The robot's red feet had kept a 900-1,300 px dark-red rim, which the material lock then locked.
  - **1, shadow.** All of these must hold:
    - the neighbour owns >= 30 % of the ring;
    - the photo is darker (ratio <= 0.85). Lighter candidates are not tested, because a highlight branch had taken a glove for a highlight;
    - the shading layer carries >= 30 % of the log-luminance step;
    - the chromaticity moved like light: no more chromatic than the neighbour (+3) with the hue within 35 deg, or a chromatic log-shift equal to the shading's within 0.08;
    - against a chromatic neighbour (C >= 20), a candidate that kept less than 0.6 of its chroma is undecidable and is kept.
  - **2, gradient.** No photo edge on the shared boundary (median |grad log Y| and |grad log chromaticity| <= 0.12), albedo cast <= 8, photo dE <= 6.
  - **3, significance.** The group is under 0.4 % and not locked, and its closest-coloured neighbour has the same body colour (lightness-normalised (a, b) within 12) with photo dE <= 6.
  - **4, crumbs.** A group of at most 150 px made only of at least 4 pieces, each at most 40 px, joins the group owning most of its ring. Example: the 58 px left of the Ducati's gold "748" edging, in 18 pieces.

  The part rim and shadow tests move single regions. The colour tests only dissolve whole groups.
- **Exemptions and guards.**
  - Tests 1-3 skip lettering, named parts, the wheel split and recovered parts (sources `'text'`, `'named'`, `'wheel'`, `'part'`), groups that are at least half decal islands, and groups mostly off the matte.
  - **Rule 1:** every part group is exempt as a candidate. A detected grip or door handle is tiny, and has its neighbour's colour on purpose.
  - A group the material lock locked is a candidate for tests 0-2, never for test 3.
  - Nothing merges into a background group or into a smaller group.
  - **Rule 2:** a part group hosts only a candidate lying at least half under its SAM mask (dilated 2 px), or its own rim.
  - **User votes.** A region the user locked or marked as background moves only into a host whose area-weighted vote agrees. The votes are in `user_flags.json`, recorded for every region of the group the choice was made on. So a pruned sliver of a locked group goes back into that group, and a sliver the user locked on its own keeps its group.
- **Backdrop crumbs.** When the fresh clustering is a product shot (background ignored by default), a tiny background group joins its closest-coloured background neighbour.
- **In the product** (the code review's probe on twelve fresh analyses): 67 moves in all, 50 backdrop crumbs, 9 shadow, 3 significance, 3 part rim, 2 crumbs. The gradient test never fired. About 0.3 s per photo on the CPU.
- **The sheen merge** (`refine.absorb_sheen`, after the pruning). A chromatic group that is lighter than an anchor and duller in the same hue (within 12 deg, at least 6 L lighter), and within the lit merge's albedo tolerance, joins the anchor. This is paint under a glossy sheen, which the photo test of `absorb_lit` rejects. The Torana's roof and boot lid (about 18k px, "Brick"), under the showroom lights, now join the body (integration).
- **The backdrop chain** (`grouping.backdrop_decisions`). A region reached from the border set through a chain of regions with matte share <= 0.15, none of them enclosed by the object, is backdrop. Pieces of the other cars in the showroom had been decided object, and had become tiny colour rows of the subject.

### 4.4 The Groups panel, edits and old jobs

- **Sections** (`sectionOf` in `web/js/panels/groups.js`; largest first in each; no headers when a job has only colour groups):
  - **Parts:** the kind's name, a green Part badge with a tooltip, and "N instances" in the meta line.
  - **Colours.**
  - **Minor:** collapsed under a divider showing its count. Each row says "next to <parent>" and has a one-click "Merge into <parent>". Minor rows are never hidden from painting.
  - **Background:** last, collapsed while the background is ignored.

  A group selected on the canvas opens its section. The selection pill says "part, N instances" or "minor", or "N parts selected" / "N colours selected" / "N groups selected".
- **Minor** (`annotate_groups`, `PANEL_RULE` 3, stored as `panel_rule` in `job.json`). A group is minor when it is not background, not a part group and holds no lettering, is under 0.4 % of the object, *and* is its parent's colour under other light. That means CIEDE2000 within 12 or, when both have a body colour of chroma >= 8, lightness-normalised (a, b) within 12.
  - The parent is the non-minor group owning most of a 5 px ring around it.
  - Size alone had made Minor a drawer of every small group, real parts included (the Ducati's 690 px gold preload adjuster).
  - It is a view hint only: a minor group is grouped, painted and edited like any other.
  - A job whose groups carry an older rule is annotated again in memory when served, without rewriting `job.json`.
- **Split.** On a part with several instances, Split splits by instance. Its tooltip says so, while the label stays "Split". A colour split of a one-colour part says "Nothing to split: Shock spring is one colour". Shares below 0.01 % read "<0.01%".
- **Regroup reproduces the analysis.**
  - The seed records, for each input region, its part tag, source and backdrop decision, plus the parts' SAM masks, the object mask and the pruning's parameters.
  - A regroup clusters the pre-snap regions again with every kind held out, then reruns the lit merge, the highlight absorb, the pruning and the sheen merge.
  - It then gives back the user's locks and background choices by area-weighted majority, and carries custom names over.
  - A seed from an older analysis regroups as that analysis did: `junk.LEGACY_OFF` turns off rules added since. It covers only the crumbs rule; see 6.9.

### 4.5 Measured

The numbers check re-ran the final state twice and reproduced these totals within the ranges shown (its runs gave 18 isolated, region isolation 0.500 / 0.505, and 213 / 214 groups).

| ten photos (194 must-be-separate parts) | baseline (reference analysis) | control (fresh, detected parts and pruning off; 2 runs) | final (2 runs) |
|---|---|---|---|
| isolated parts (strict) | 6 (0.031) | 6 (0.031) | 18 (0.093) |
| kind-level isolation, of 133 | 2 (parts-lane control) | – | 27 |
| region isolation | 0.490 | 0.490-0.495 | 0.500-0.505 |
| purity / coverage | 0.181 / 0.887 | 0.179-0.180 / 0.887-0.888 | 0.283-0.286 / 0.887-0.889 |
| merged / fragmented | 185 / 27 | 185-186 / 27 | 169-170 / 25-26 |
| by-eye junk left, of 16 (severity, of 25) | 16 (25) | 15 (23) | 10, 1 partial (15) |
| real small groups lost, of 17 | – | 0 | 3 |
| automatic junk groups | 3 (4,537 px) | 3 (4,537 px) | 1 (1,581 px: the Jaguar's windscreen reflection) |
| object slivers | 7 (8,041 px) | 7 (8,041 px) | 6 (9,769 px); 4 of them are detected parts (Grip, Footpeg, Badge, Fuel cap) |
| tiny groups (on the object) | 79 (30) | 79-82 (29) | 42-43 (30) |
| groups (background, locked) | 230 (105, 8) | 229-230 (106-107, 8) | 213-216 (77-79, 5) |
| panel rows: colours / minor | – | 114 / 9 | 97-98 / 1 |
| stamped masks (true / other part / unlabelled) | – | – | 51 (40 / 5 / 6) |
| twelve photos: isolated of 221 / region isolation / kind-level of 148 | – | 8 / 0.493-0.498 / – | 26 / 0.507-0.511 / 35 |
| twelve photos: calipers found / strictly isolated (7 visible) | – | 0 / 1 | 6 / 2 |

The strict count is 18, not the lanes' 20. Each bike's second wheel is now found, so its two tyres share one "Tyres" group until a Split. The kind-level count rises.

Per photo: the baseline is the reference analysis (for the two added cars, the fresh control); the final is the integration's run `fresh_j`. Its twin `fresh_k` differs only in two group counts.

| photo | isolated: baseline -> final | region isolation | by-eye junk (severity) | groups |
|---|---|---|---|---|
| motorcycle_1 | 1 -> 6: adds front caliper, shock spring, seat, rear sprocket, right grip | 0.61 -> 0.61 | 4 (6) -> 4 (6) | 25 -> 31 |
| motorcycle_2 | 2 -> 3: adds seat and rear disc (named "Sprocket"); loses the front caliper, now in "Brake calipers" with the rear | 0.52 -> 0.55 | 1 (2) -> 0 | 12 -> 17 |
| car_red_sports_1 | 0 -> 3: front door handle, rear bumper, exhaust tip | 0.62 -> 0.69 | 1 (3) -> 0 | 43 -> 28 |
| car_red_sports_2 | 0 -> 0 | 0.30 -> 0.30 | 0 -> 0 | 20 -> 21-23 |
| car_classic_1 | 1 -> 3: adds grille, mirror head | 0.39 -> 0.39 | 1 (2) -> 1 partial (2) | 16 -> 16 |
| car_classic_2 | 1 -> 2: adds grille | 0.30 -> 0.30 | 5 (7) -> 4 (6) | 31 -> 29 |
| bicycle_1 | 0 -> 0 | 0.25 -> 0.25 | 0 -> 0 | 16 -> 16 |
| sneakers_2 | 0 -> 0 | 0.08 -> 0.08 | 3 (4) -> 0 | 22 -> 15 |
| robot_toy_2 | 0 -> 0 | 0.67 -> 0.67 | 1 (1) -> 1 (1) | 29 -> 27 |
| gundam_rx78_rg | 1 -> 1 | 0.55 -> 0.58 | 0 -> 0 | 16 -> 14 |
| car_alpine_1 / car_corvette_1 | 2 -> 5 / 0 -> 3 (the Corvette's caliper, front rim, mirror) | 0.59 -> 0.65 / 0.40 -> 0.40 | not catalogued | 25 -> 30 / 26 -> 27 |

**evalkit** (`ours@quick`, 6 photos; PACO, 21 images). The numbers check recomputed the regions stage live, because the integration's run reused cached partitions from an earlier integration, and 4 of 27 of them differ. The live column is the verified one.

| metric | before (parts-lane control) | round 6 (cached) | final, cached (integration) | final, live (numbers check) |
|---|---|---|---|---|
| ASA / ASA parts / UE | 0.945 / 0.866 / – | 0.945 / 0.865 / 0.109 | 0.947 / 0.870 / 0.105 | 0.947 / 0.871 / 0.106 |
| best IoU / R@.5 | 0.674 / 0.747 | 0.678 / 0.747 | 0.682 / 0.753 | 0.680 / 0.747 |
| achievable recall / thin-part recall / purity | 0.820 / 0.474 / 0.962 | 0.820 / 0.474 / 0.962 | 0.831 / 0.526 / 0.967 | 0.831 / 0.526 / 0.968 |
| boundary F@2 / P@2 / F@4 | 0.790 / – / – | 0.790 / 0.828 / 0.850 | 0.780 / 0.802 / 0.841 | 0.775 / 0.792 / 0.836 |
| PACO ASA / F@2 / best IoU / R@.5 / achievable recall | – / 0.651 / 0.337 / 0.327 / 0.388 | 0.922 / 0.651 / 0.348 / 0.343 / 0.408 | as round 6 | 0.921 / 0.648 / 0.347 / 0.339 / 0.404 |

Without the disc look, cached F@2 is 0.788, so about 0.008 of the drop is the disc's outline.

**Behaviour checked end to end.** These runs used private servers and private data dirs.
- **Ducati part rows:** Wheel rims (2 instances), Tyres (2), Exhausts (2), Seat, Sprocket, Brake disc, Shock spring, Grip, Brake caliper. 30 groups, no Minor rows (numbers check). The Corvette's "Brake caliper" is 1,595 px, isolated at purity 0.87 and coverage 0.99. The BMW and the Alpine each show a two-instance "Brake calipers" row.
- **Painting a part alone.** Pixels changed more than 8 px away from the part (numbers check):

| part painted alone | far pixels changed |
|---|---|
| shock spring | 3 |
| Ducati caliper | 16 |
| Corvette caliper | 38 |
| Ducati tyres | 27 |
| Corvette tyre | 4 |
| seat | 40 |
| Ducati rims | 505 |

  99-100 % of each part's core changes.
- **Edits** (code review, on copies of six fresh jobs, 6 of 6 on each check; regroup parity 12 of 12):
  - A regroup at the analysis's options changes nothing.
  - Locking every group and then regrouping keeps the partition and the locks.
  - The smallest group, locked alone, survives a regroup.
  - A colour split followed by a merge and a regroup gives one group per kind.
  - A part region moved into a colour group returns to its part on regroup.
  - Split pieces keep their part tags, and Merge restores the plural name.
- **Split by instance** (numbers check): "Wheel rim (front)/(rear)", "Tyre (front)/(rear)", "Exhaust (rear)/(front)". Rendering both halves differs from the parent by 0 px, and merging back also gives 0 px difference.
- **Old jobs** (numbers and UI checks): a pre-parts fixture copy and a round-6 copy both open without errors, their `job.json` md5 is unchanged, and painting works. A job stored with 9 Minor rows is served with 1.
- **Subject check:** the coupe behind the Torana is a background group ("Cherry"), and the Torana's paint group is "Red" (UI check). In the coupe's box, 54 of 12,135 px change under a navy repaint of the Torana (integration).
- **Halo.**
  - The fixtures' stored jobs are unchanged: 5,573 px in total, identity 0.
  - Fresh navy repaints of the ten photos score 55,700-55,775 px in the paint family (round 6: 55,068-55,103). All of the rise is the Torana: the coupe, now red on purpose, sits inside the 5 px ring the harness scores as old hue.
  - On twelve photos, the Alpine's 12 -> 435 is the harness painting the number plate's blue band (12 deg from the navy target) as the "most chromatic group" (integration).
- **Cost** (integration, except where noted):
  - Balanced analysis: 10.6-11.4 s per photo with warm models, 20.5 s with cold models. The UI check measured 31 s and 17 s from the Home strip with cold models.
  - Part step: 0.4-1.2 s per image. The wheel look costs about 0.5 s on a bike and 0.04-0.1 s on a car.
  - Pruning: about 0.3 s on the CPU.
  - VRAM: OWLv2 adds 0.8 GB while loaded. Peak 10.91 GB reserved in the fresh runs (the numbers check measured the same).
  - Full-resolution export: 1.2-4.3 s.

### 4.6 The checks

**Numbers check: failed (1 major, 2 minor).** What it did:
- re-ran the unit tests: 421 pass;
- ran two fresh in-process twelve-photo analyses into a private data dir and scored them with a copy of the integration's scorer; every total matches within the ranges above;
- checked the fixture halo (identical) and the fresh halo (within 50 px);
- on a private server running the working tree, analysed the Ducati and the Corvette at Balanced and exercised:
  - regroups: all 9 parts survive at 6, at 12 and at the default, and the default restores the identical partition;
  - lock then regroup;
  - split by instance and merge;
  - a colour split of the spring (a no-op);
  - part-alone repaints;
  - two copied old jobs.

Major: the Ducati's brake disc (6.1). Minors:
- evalkit came from cached partitions, so the live numbers are slightly lower, and "PACO unchanged" was never measured live;
- the far-change list left out the rims (505 px) and the Corvette caliper (38 px, above the reported 0-31).

**Code review: passed (6 minor).** What it confirmed:
- regroup parity and the edit flows above;
- the vote logic only restricts moves;
- `refresh_panel_view` changes memory only;
- every model-backed step catches errors, logs them, frees CUDA and keeps the partition;
- Fast skips all of them;
- old jobs without a seed fall back to `grouping.regroup`, which keeps part kinds apart;
- the gates are named constants;
- the diff has no private paths, IPs, attribution lines or debug prints.

The minors are listed in 6.9.

**UI check: failed (1 major, 2 minor).** Headless Chromium against a private server running the working tree:
- two samples analysed from the Home strip;
- the Groups panel in sections: the Ducati shows Parts 9, Colours 18 and Background 4 (collapsed); the Torana shows Parts 5, Colours 10 (the paint "Red" at 17 %) and Background 13, including the coupe;
- the pill painting always opened the right group's picker;
- Split by instance gives position names, and the mapping follows both instances;
- Auto-regroup keeps every part and the mapping;
- a lock wins over its mapping;
- merge works;
- exports: a 3067x2045 JPG in 6.8 s that matches the preview, with the toast covering 0 px² of the result card;
- the two old copies open unchanged and paint;
- dark theme, and 390 px width with no horizontal scroll;
- 0 console errors, 0 HTTP errors, 0 server errors.

Neither fresh job had Minor rows, so the collapsed Minor section was not exercised. Major: the Torana's rear rim (6.2). Minors: 6.8 and 6.10.

## 5. The judges' verdict

Three judges, each with one lens, scored the six human comparison sheets in `docs/comparison/parts/`: `motorcycle_1.jpg`, `motorcycle_2.jpg`, `car_red_sports_1.jpg`, `car_classic_1.jpg`, `bicycle_1.jpg` and `gundam_rx78_rg.jpg`.

**How the sheets were made.** Each sheet is one fresh in-process analysis through `recolor.pipeline.analyze`, in a private data dir. CURRENT is the same run's partition, captured just before `_detect_parts`, so SAM and the intrinsic layers are the same as NEW's. It was then grouped again with the detected parts, the subject check and the pruning switched off. The sheen merge has no switch and runs in both.

**What each sheet shows:**
- **Row 1:** ORIGINAL, CURRENT groups, NEW groups. Every group is filled with its albedo colour and the background is faded. In CURRENT, the junk that the pruning folds is outlined in yellow. In NEW, part groups are outlined and labelled.
- **Row 2:** the paint. Navy goes on `refine.paint_family`, and red on every shock spring and caliper group plus one more part (yellow when that part is already red). CURRENT has no part groups, so red goes on the groups holding at least 30 % of the part.
- **Close-ups** at 1.4-3x: springs and calipers first, then a shadow that was its own group, then small parts.

Group counts, CURRENT / NEW: Ducati 26 / 31 (9 parts), BMW 11 / 16 (7), Torana 39 / 28 (5), van 16 / 16 (3), bicycle 16 / 16 (2), Gundam 16 / 14 (1). No lever was detected on any of the six.

| lens | NEW | CURRENT |
|---|---|---|
| Personalisation parts (springs, calipers, rims, levers, mirrors) are their own groups, and nothing real is lost | 7 | 2 |
| No junk groups (shadows, reflections, gradients), and no seams or speckle in the paint | 7 | 3 |
| Overall: the groups a user expects, in the order expected | 6 | 2.5 |
| mean | 6.7 | 2.5 |

All three judges ranked NEW first and called it better than CURRENT.

**Lens 1 (parts).**
- In NEW, every named part found is its own group, and painting it changes only that part: the Ducati's spring and front caliper, the BMW's front and rear calipers, the van's mirrors, the Gundam's head, the bicycle's pedal and bottle, the Torana's door handle and exhaust tip.
- In CURRENT, painting a part repaints the large group holding it:
  - the spring takes the gold frame tubes with it;
  - the BMW's calipers and rims turn most of the bike red;
  - the van's mirror repaints the whole van;
  - the pedal repaints both bikes' frames and wheels and a rider's kit;
  - the Gundam's head repaints all the white armour;
  - the Torana's door handle repaints the whole body.
- NEW loses points for incomplete rims, rims that swallow discs, the parts it misses, and a few real small groups it absorbed. CURRENT keeps some of those small groups.

**Lens 2 (junk and paint).**
- NEW's body paint shows no seams on the Ducati, the BMW, the Torana or the Gundam.
- The shadow and crumb groups CURRENT marks are gone with no leftover patch: the Ducati's 1,436 px "Gray 5" sliver, the Torana's "Maroon" flare shadow, the van's "Graphite 4" sill strip, the Gundam's "Black 2" backdrop crumb.
- Part paint stays on the part. What remains in NEW is local speckle and bleed at a few part edges.
- CURRENT's grouping has little junk on the object, but its paint fails because parts sit inside huge groups. Unfolded fragments also streak the paint, such as the Ducati's bronze crumbs in the lettering.

**Lens 3 (overall).**
- In NEW, the named parts are groups a user can pick.
- Held back by:
  - incomplete rim, exhaust and bumper groups;
  - two wrong names;
  - single instances of repeated parts;
  - coarse groups on the van, the bicycle and the Gundam;
  - the wrong paint family on two photos.

**Defects that remain in NEW, per sheet** (merged across the three judges):

- **Ducati.**
  - "Brake disc" is patchy pink islands, not a ring. The carrier and part of the rotor sit in "Wheel rims", so the rotor turns pink with the rims.
  - The rear rim's lower spoke and the hub area near the sprocket carrier stay gold, and the front rim keeps gold patches on some spokes and at the hub.
  - A second gold caliper, behind the disc, is not found (it is not in the reference either).
  - Pink dots on the fairing fasteners and an orange fuel-cap patch stay unrecoloured, as in CURRENT.
- **BMW.**
  - The front disc is inside "Wheel rims" and turns fully red, while the bottom of the front rim's lip stays black.
  - On the rear rim, only the upper-left band and a few spokes turn red, which leaves a seam against the black spokes.
  - Pink-red blobs sit on the black knob at the fork top and on the swingarm behind the silencer, and pink tints appear in the vent slots and the tank-side mesh.
  - "Exhaust" covers only the two collector pieces under the engine: not the collector box, the silencer, or the left end of the heat-stained header. Navy speckle bleeds onto the copper header.
  - "Footpeg" sits on the fork foot and axle clamp; the real footpegs are not found.
- **Torana.**
  - The rear rim's polished lip stays silver, while the front rim is painted to its lip.
  - The amber tail lamp's group is absorbed into the body.
  - Only the front door handle is a part: the rear handle and the lock stay unlabelled. Only the rear bumper is "Bumper". The mirror stays in the glass group.
  - The coupe seen through the side and rear windows takes the navy.
- **Van.**
  - "Fog lamp" is really the round chrome headlamp; the actual amber fog lamp is not found. The group holds the lens, a thin rim and a detached sliver of the chrome shell, plus a chain of dot crumbs. The lower-rear housing is left as a hole in body grey, and red bleeds as a ragged fringe onto the black mount.
  - "Mirrors" holds the heads only, not the gold arm, and the door handle is not detected.
  - The "Grille" outline runs down over the licence plate.
  - The tyres and wheels are inside the body group.
  - The paint family is the gold lettering and bumper, not the navy body.
- **Bicycle.**
  - Only one pedal and the white bike's bottle are found.
  - "Bottle" trails stray dots along the rider's calf and the frame.
  - Rims, discs, calipers, levers and the crankset are not parts, and both bikes' frames and wheels plus a rider's black kit are one "Graphite".
  - The paint family is the riders' skin.
- **Gundam.**
  - "Head" is the only part: no V-fin, rifle, shield, arm, leg or foot parts (the rifle and shield remain separate groups).
  - The red crest and chin are outside "Head", so they take the paint family's colour.
  - Two small grey mechanical pieces at the right shoulder joint are merged into the neighbouring grey.
  - The white shield rim is merged with all the white armour.

## 6. Remaining gaps

**6.1 The Ducati's brake disc (major, numbers check, open).**
- **What is wrong.** SAM's disc answer has 19 components. It covers the lower band of the rotor plus blotches, while the rest of the rotor and the black carrier stay in "Wheel rims".
  - In both of the check's fresh runs, the reference front disc is 0.56 "Brake disc" and 0.44 "Wheel rims". 15 % of the front rim sits in "Brake disc", and 25 % of the front caliper in "Wheel rims".
  - The labels already interleave across the rotor in the stamped partition, so neither the snap nor the pruning caused it.
- **What the user sees.** At full resolution, painting "Wheel rims" red, "Brake disc" red, or the rims red and the disc green each gives a blotchy camouflage across the rotor (`scratch/integration5/checks/r7num/e2e/ducati_rotor_*.jpg`). In round 6 the disc sat inside the rim and turned one even colour.
- **A misleading number.** The integration's "110 of the disc's 1,571 core px change when the rims are painted" measured only the disc group's own core.
- **The check's fix.** Either of:
  - take SAM's disc answer as one filled annulus inside the rim ellipse (fill holes, close, keep the component holding the ring, or take the rotor band from the ring geometry `wheels.py` already fits);
  - or reject a disc mask this fragmented, with a solidity or fragmentation gate, and leave the disc in the rim as before.

  Add a unit test that fails on a speckled disc, and an end-to-end check of the rotor's share in the rim group.

**6.2 Rims that do not hold their lip or spokes (major, UI check, open).**
- **On the Torana.** "Wheel rims" covers the front rim whole, polished lip included. It leaves the rear rim's outer lip and parts of its spokes in "Khaki 2", a shiny colour row.
  - In the ring at 1.0-1.15 of the rear rim's radius, "Khaki 2" holds 2,024 px and "Wheel rims" only 197. The rear rim's best group is still "Khaki 2" (coverage 0.53).
  - Painting the rims leaves the rear lip silver and the spoke edges spotted. The same happens after a split by instance or an Auto-regroup (`scratch/integration5/checks/ui/r7/23_coupe_rims_sheet.png`).
- **The same family elsewhere:**
  - The BMW's "Wheel rims" covers 0.64 of the front rim and 0.74 of the rear, and holds 0.97 of the front disc. SAM's disc answer took in the fork foot and failed the gates. So a rim repaint paints the front disc, and leaves the rear's lower and right spokes black.
  - The Ducati's rear rim leaves a lower spoke and the hub gold, and its white lip turns pink with the rim, because the lip is part of the rim mask (integration).
  - The Ducati's rims, painted alone, change 505 px more than 8 px away (in Graphite, Sprocket, Silver 2 and Exhausts; max 25 levels).
- **The check's fix.** One rule for whether a rim includes its lip, applied to every instance: pull the pixels of a same-kind colour group lying inside the rim's disc into the rim, or trim the lip on both. Then check that both instances paint alike.

**6.3 Parts not found.**
- **Not found:**
  - levers, on every photo;
  - the bicycle's rim brake, levers, rims and crankset (its thin, motion-blurred wheels pass no gate);
  - sneaker panels, laces and sole;
  - Gundam armour ("Head" is the whole head, coarser than the reference helmet);
  - the robot's arms;
  - second instances (the black bike's pedals and bottle);
  - the BMW's collector box, silencer and real footpegs;
  - the Torana's mirror, rear door handle and front bumper;
  - the van's door handle and mirror arm;
  - the second gold caliper behind the Ducati's disc.
- **Calipers.** 6 of the 7 visible calipers in the twelve-photo set are found: all but the bicycle's rim brake. On the other four cars none is visible or resolvable. Only 2 of 7 are strictly isolated, because the two calipers of one vehicle share one group until a split. The Alpine's front caliper is covered at 61 % (integration).
- **Photos that gain nothing.** By strict isolation, five of the ten photos gain nothing: the Boxster, the bicycle, the Fila, the robot and the Gundam.
- **The next lever is the detector, not the grouping:** SAM 3 concept prompts if the owner accepts Meta's gated licence, or OWLv2 queried with image exemplars or fine-tuned on caliper, lever and spring crops.

**6.4 Wrong names and stray part pixels.**
- **Wrong names.** 5 of the 51 stamped masks lie mostly on another part:
  - the BMW's rear disc, named "Sprocket";
  - the van's headlamp, named "Fog lamp";
  - the Jaguar's side lamp, named "Fuel cap";
  - the Gundam's whole head, named "Head";
  - a rim that holds its disc.

  6 more lie on unlabelled pixels, including the BMW's "Footpeg" on the fork foot. A wrong name is a misnamed row, not a lost part: the row is unlocked and made of whole instances, so a rename or a merge undoes it.
- **Stray pixels** (all of these were flagged by the judges, 5.1-5.3):
  - part fragments on unrelated metal on the BMW;
  - the van "Fog lamp"'s ragged fringe;
  - the van grille's outline over the licence plate;
  - the "Bottle"'s trail of dots;
  - navy speckle on the BMW's copper header.
- **Far change.** The BMW's exhaust, painted alone, changes about 440 px more than 8 px away: the heat-tinted collector, max 32 levels (integration).

**6.5 Junk left.** 10 of the 16 catalogued groups remain (1 partial), severity 15. Without the misfiled Öhlins logo it would be 9 groups, severity 12. What stays, and why:
- **The Jaguar's reflections of other cars** (four groups). They sit at photo dE 11-47 and colour casts 10-42 from their host. The residual layer carries -0.1 to 0.4 of the difference, the same range as real badges. The albedo holds the reflection, so no photometric test here separates them.
- **The Ducati's two cavity shadows.** They were stamped as small distinct parts and are exempt, because exempting islands is what keeps real decals safe.
- **The Ducati's sprocket hole.** It shows another surface, and its chroma rises from 2 to 15.
- **The robot's rim light.** It is lighter than its host, and the highlight branch is off.
- **Part of the van's sill band.**

The large splits of one neutral paint are unchanged:
- the Jaguar body's best group still covers only 0.56 of it;
- the Fila's tongue is still split 52 / 47 between two greys;
- the bicycle's fork is split 46 / 39 / 15 %.

The pruning only looks at groups under 0.4 % of the object. Above that, the shadow test cannot tell a dark neutral material from a shadow on neutral paint: at 2 % it took a real 21.8k px dark group, and at 5 % pieces of the RX-78's rifle. And `absorb_lit` never merges neutral pairs.

**6.6 Real small groups lost, and a lamp.** 3 of the 17 catalogued real small groups are lost:
- the orange drop-shadow fringe of the Ducati's "748" (58 px in 18 pieces). The crumbs rule folds it on purpose, because it left an orange fleck under a navy repaint;
- the BMW's far gold preload adjuster, which shares a locked group with its twin on the near fork;
- the paper sign seen through the Torana's C-pillar window. It belongs to the coupe behind and is now background with it.

Also:
- The Torana's amber tail light is half (0.49) in the body's group in the final runs, and gone from the sheet's NEW.
- Two small grey pieces at the Gundam's shoulder joint are merged into the neighbouring grey.

**6.7 The shadow test has no darkness floor on a neutral host.** The chroma floor protects only chromatic hosts. The Fila's lug gaps (shadowed ground seen through the tread) went into the white shoe at a photo ratio of 0.01-0.017. This is invisible because they are near black, but wrong in meaning. The missing rule is a floor on the luminance ratio for neutral hosts.

**6.8 Parts against the paint.**
- **Painted parts leave the paint family.** A painted part detected as a kind leaves the paint family: the robot's feet go from 1.000 to 0.001 in it. A body repaint leaves them red, although they can be painted on their own.
- **Body-coloured parts.** The Torana's body-coloured door handle is left alone when the body is repainted, so its recess keeps the old paint's reflection. This needs a product decision: either body-coloured parts follow the paint unless mapped on their own, or a "paint with body" option. The stack lane's opt-in "Paint with <parent>" was not shipped.
- **Windows.** Whatever the subject's windows show is painted with the subject: the coupe seen through the Torana's glass takes the navy.
- **Paint-family defaults.** The default paint family is not always what a user expects, because `refine.paint_family` takes the largest chromatic group above chroma 18. Examples: the van's gold lettering and bumper instead of its navy body, the riders' skin on the bicycle, 657 px on the Gundam.
- **Far change from the paint itself.**
  - Painting the Ducati's "Red" changes 197 px of "Graphite" more than 8 px away: red reflections in the windscreen and mudguard, max 35 levels (UI check). It also changes 8 px of "Silver 2".
  - The Corvette's navy repaint changes 2,443 px far away (max 131 levels). This is the pre-existing coupling to its stripes and yellow interior (integration).

**6.9 Code-review minors (open).**
- **(a) The crumbs rule's exemptions.** It checks only that a group is not a part and holds no lettering. It skips the other exemptions: the `'wheel'` and `'part'` sources, the island share, the matte share and locks. So a locked island group of crumbs with source `'part'`, `'wheel'`, `'prompt'` or `'small'` folds into the paint; this was reproduced on a synthetic scene. On twelve fresh analyses the rule fired twice, both times on the intended 58 px fringe.
- **(b) The docs contradict the crumbs rule.** README says lettering, decals, recovered parts and detected parts are never folded, and `junk.py` says "four tests" and "exempt from every test". ARCHITECTURE.md is accurate.
- **(c) `junk.LEGACY_OFF` covers only the crumbs rule.** 90 seeds on disk (8 of them in `data/jobs`) predate the part-rim rule, so their regroups now run a rule their analysis never ran. Measured on copies of the 8: 0 part-rim moves.
- **(d) Split, lock, regroup.** Split a part by instance, lock one instance, then Auto-regroup. The regroup rebuilds one group per kind, and the area-majority vote either locks the whole kind or loses a minority lock. The split is not restored.
- **(e) Test coverage.** `find_discs` has no unit test. `subject.cut_off` is covered only through a monkeypatched test.
- **(f) Islands on regroup.** `regroup_refined` drops the pruned islands, so `islands.npy` is not rewritten when a regroup at other options folds a different decal island.

**6.10 UI minor (open).** Auto-regroup at 6 colours spends half the budget on specks: Black 0.05 %, Copper 0.05 %, and a locked Crimson 0.01 %. Meanwhile mid-size colours merge into "Gray 2" (55 regions). Tiny auto-locked or protected groups should not count against `max_groups`, and the UI should say why a group the user never locked shows as locked (it currently shows "Crimson" and "Chocolate" that way).

**6.11 Measurement limits.**
- F@2 drops from 0.790 to 0.775 on the live evalkit run, and about 0.008 of that is the disc's outline.
- The fresh-halo rise and the Alpine's 435 px are harness effects (4.5).
- Every threshold was tuned in-sample.
- The strict metric counts per-kind groups as merged.
- The by-eye catalogue covers only the baseline's groups; 14 of the stack's Minor rows were never judged.
- Runtimes were taken on a shared card.

**6.12 Old jobs and the live app.** Jobs analysed before this round keep their labels and groups (no parts, no pruning) until they are re-analysed. Only their Minor rule is refreshed, in memory, and the stored flags are rewritten on the next edit. The owner's app has not been run on this code: every check used a private server and data dir.

**Next levers, in order:**
1. The disc (6.1) and one rim rule (6.2), both inside the shipped mechanism.
2. A darkness floor for neutral hosts (6.7), and the crumbs rule's exemptions (6.9 a).
3. The product decision on body-coloured parts (6.8).
4. A stronger part detector for calipers, levers, sneaker parts and kit parts (6.3), which is where the remaining part recall is.

## 7. Bibliography

Every candidate the two surveys surfaced, the models the shipped stages run, and the evaluation data. Licences are as the surveys recorded them from repository licence files and model cards. "Not run" means surfaced and not tried, for the stated reason.

**Part detectors and part segmentation**
- **OWLv2.** Minderer, Gritsenko, Houlsby, Scaling Open-Vocabulary Object Detection (NeurIPS 2023). https://arxiv.org/abs/2306.09683 ; weights https://huggingface.co/google/owlv2-large-patch14-ensemble (Apache-2.0).
  - Measured (`parts`): the detector that finds small parts.
  - Shipped (`partdetect.py`, revision `95e2693`, fp16, score >= 0.3).
  - No caliper above 0.2 at photo scale on the bikes, hence the wheel second look. Its image-exemplar mode was surveyed but not run.
- **Grounding DINO.** Liu et al. (ECCV 2024). https://arxiv.org/abs/2303.05499 ; checkpoint `IDEA-Research/grounding-dino-base` (Apache-2.0). Measured (`parts`): best on wheels and seats, misses small parts, and its wheel boxes cost region isolation. Rejected.
- **Florence-2.** Xiao et al. (CVPR 2024). https://arxiv.org/abs/2311.06242 ; weights https://huggingface.co/florence-community/Florence-2-large (MIT; the survey cites https://huggingface.co/microsoft/Florence-2-large). Measured (`parts`):
  - per-phrase `<OPEN_VOCABULARY_DETECTION>` rejected: tile-sized boxes, and confidences that do not separate right from wrong;
  - `<CAPTION>` shipped as the object-class switch (10 of 10 correct), next to the round-3 OCR and phrase grounding.
- **SAM 3 / SAM 3.1**, Segment Anything with Concepts (Meta, 2025). https://arxiv.org/abs/2511.16719 ; code https://github.com/facebookresearch/sam3 ; `Sam3Model` is in `transformers` 5.17. Weights `facebook/sam3` / `facebook/sam3.1` are gated on Hugging Face under the SAM License (Meta custom: commercial use allowed, licence copy shipped, military / nuclear / ITAR restrictions).
  - Not run: gated, and the owner must accept the licence; it is not to be bypassed.
  - The parts survey's first choice (masks, presence head, exemplars). Meta reports weak zero-shot on fine-grained out-of-domain terms.
- **MM-Grounding-DINO** (arXiv 2401.02361) and **LLMDet** (CVPR 2025). https://arxiv.org/abs/2501.18954 ; https://huggingface.co/docs/transformers/model_doc/mm-grounding-dino (Apache-2.0). Not run. The survey's best open-weight rare-class detector.
- **Grounding DINO 1.5 / 1.6 Pro and Edge** (IDEA Research, 2024). https://arxiv.org/abs/2405.10300 ; API client https://github.com/IDEA-Research/Grounding-DINO-1.5-API (Apache-2.0 client; weights proprietary, API only). Not run: every customer photo would leave the machine.
- **Rex-Omni**, Detect Anything via Next Point Prediction (2025). https://arxiv.org/abs/2510.12798 ; code https://github.com/IDEA-Research/Rex-Omni (IDEA License 1.0 under the Qwen research terms; commercial use doubtful). Not run.
- **VLPart.** Sun et al., Going Denser with Open-Vocabulary Part Segmentation (ICCV 2023). https://arxiv.org/abs/2305.11173 ; code https://github.com/facebookresearch/VLPart (MIT, with CLIP / Detic / dino-vit-features portions under their own licences; archived November 2024). Not run: a detectron2 build on torch 2.14 / CUDA 13 is unverified, and it has no motorcycle category.
- **PartGLEE.** Li et al. (ECCV 2024). https://arxiv.org/abs/2407.16696 ; code https://github.com/ProvenceStar/PartGLEE (no licence file, which blocks integration). Not run.
- **PartCLIPSeg**, Understanding Multi-Granularity for Open-Vocabulary Part Segmentation (NeurIPS 2024). https://proceedings.neurips.cc/paper_files/paper/2024/file/f7f47a73d631c0410cbc2748a8015241-Paper-Conference.pdf ; code https://github.com/kaist-cvml/part-clipseg (MIT). The OV-PARTS benchmark (NeurIPS 2023): https://github.com/OpenRobotLab/OV_PARTS (no licence stated). Not run: pinned to torch 2.2.2 / mmcv 1.7, 352 px input, generic vocabularies.
- **Semantic-SAM** (2023). https://arxiv.org/abs/2307.04767 ; code https://github.com/UX-Decoder/Semantic-SAM (MIT per the junk survey; round 3 recorded no licence file). Not run.
- **PACO.** Ramanathan et al., Parts and Attributes of Common Objects (CVPR 2023). https://arxiv.org/abs/2301.01795 ; https://github.com/facebookresearch/paco (code MIT; the parts survey records the annotations as CC0, the junk survey as CC-BY-NC "to check"; 41 of the 63 local images are NC and / or ND, so research use only). The source of the vocabulary's names, and the `paco@quick` evalkit set.
- **Carparts-Seg** (Ultralytics / Roboflow; 3,833 images, 23 classes). https://docs.ultralytics.com/datasets/segment/carparts-seg (CC BY 4.0). Roboflow's Florence-2 LoRA fine-tuning recipe: https://blog.roboflow.com/fine-tune-florence-2-object-detection/ (mAP50:95 0.52, against 0.9 for YOLOv8-S on the same data). Not used: no caliper, spring or lever labels.
- **FG-OVD**, The devil is in the fine-grained details (CVPR 2024). https://arxiv.org/abs/2311.17518 (arXiv, CC BY). Design evidence: open-vocabulary detectors find the noun but fail on colour and material attributes. So the vocabulary prompts bare nouns ("brake caliper", never "gold caliper"), and colour stays the grouping's job.

**Shadows, lighting and region merging (the junk survey)**
- **Guo, Dai, Hoiem**, Paired Regions for Shadow Detection and Removal (CVPR 2011; PAMI 2013). http://dhoiem.cs.illinois.edu/publications/cvpr11_shadow.pdf ; MATLAB demo https://archive.org/details/shadow_code_10.1109CVPR.2011.5995725 (no licence stated). Not run as such. The shipped shadow test follows its premise (judge a region against a neighbour at the seam, never alone), without its per-channel ratios or classifier.
- **van de Weijer, Gevers, Geusebroek**, Edge and corner detection by photometric quasi-invariants (PAMI 2005). https://lear.inrialpes.fr/people/vandeweijer/papers/pami2005.pdf ; **Gevers and Stokman**, Classifying color edges in video into shadow-geometry, highlight, or material transitions (IEEE TMM 2003). https://staff.fnwi.uva.nl/th.gevers/pub/GeversMM03.pdf ; third-party code https://github.com/eokeeffe/quasi_invariant-features (licence not checked). Not run this round; round 3 found the invariant reads a saturation change as a highlight.
- **Maxwell, Friedhoff, Smith**, A Bi-Illuminant Dichromatic Reflection Model for Understanding Images (CVPR 2008). https://www.cs.colby.edu/maxwell/papers/pdfs/Maxwell-CVPR-2008.pdf (no code; the spectral-ratio ideas are patented by Tandent in the US, the US20100142805A1 family). Not run. The shadow test's cast branch accepts a coloured shadow on the same premise (the shift is the light's), but takes the light's colour from the intrinsic shading instead of estimating it from pairs.
- **Finlayson, Hordley, Lu, Drew**, illuminant-invariant log-chromaticity and shadow removal (PAMI 2006). https://www.cs.sfu.ca/~mark/ftp/Pami06/pami06.pdf ; the calibration-free invariant direction by entropy minimisation (Finlayson, Drew, Lu, ECCV 2004 / IJCV 2009): https://link.springer.com/article/10.1007/s11263-009-0243-z (patented in the US, US7751639). Not run this round; round 3 found it is noise on saturated paint.
- **Tappen, Freeman, Adelson**, Recovering Intrinsic Images from a Single Image (PAMI 2005). https://people.csail.mit.edu/billf/publications/Recovering_Intrinsic_Images.pdf (no code). The two-threshold colour Retinex of **Grosse, Johnson, Adelson, Freeman** (ICCV 2009), code and data: http://www.cs.toronto.edu/~rgrosse/intrinsic/ . Not run.
- **Kovacs, Bell, Snavely, Bala**, Shading Annotations in the Wild (CVPR 2017). https://openaccess.thecvf.com/content_cvpr_2017/papers/Kovacs_Shading_Annotations_in_CVPR_2017_paper.pdf ; code https://github.com/kovibalu/saw_release (Caffe era; check the repository licence). Not run.
- **Felzenszwalb, Huttenlocher**, Efficient Graph-Based Image Segmentation (IJCV 2004). https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf ; code https://cs.brown.edu/people/pfelzens/segment/ (GPL); `skimage.segmentation.felzenszwalb` (BSD-3). Not run. The survey's cheapest proposal, a size-adaptive linkage threshold in `cluster_colors`, was not tried; the pruning's fixed size gate (0.4 % of the object) takes its place.
- **Nock, Nielsen**, Statistical Region Merging (PAMI 2004). https://www2.sonycsl.co.jp/person/nielsen/infogeo/FrankNielsen/Journals/JOURNAL/2004-J-TPAMI-StatisticalRegionMerging.pdf ; third-party https://github.com/ka-petrov/LibSRM (check its licence). Not run.
- **Zhu, Yuille**, Region Competition (PAMI 1996). https://www.cnbc.cmu.edu/~tai/papers/region_competition.pdf ; http://vcla.stat.ucla.edu/old/Segmentation/Region_competition/region_competition.htm ; with the piecewise-constant Mumford-Shah merging of Koepfler, Lopez, Morel (SIAM J. Numer. Anal. 1994). Not run.
- **Arbelaez, Maire, Fowlkes, Malik**, Contour Detection and Hierarchical Image Segmentation (PAMI 2011). https://people.eecs.berkeley.edu/~malik/papers/arbelaezMFM-pami2010.pdf ; **Calderero, Marques**, information-theoretic region merging with a partition significance index (IEEE TIP 2010). https://upcommons.upc.edu/handle/2117/7488 . Not run. The pruning's gradient test (no photo edge on the shared boundary) is the simplest form of UCM's boundary saliency, and it made 0 of the 67 moves on twelve fresh analyses.
- **Klinker, Shafer, Kanade**, A Physical Approach to Color Image Understanding (IJCV 1990). https://www.ri.cmu.edu/pub_files/pub3/klinker_g_1990_1/klinker_g_1990_1.pdf ; **Shafer**, the dichromatic reflection model (Color Res. Appl. 1985). The physics behind `grouping.normalise_lab` and `materials.absorb_highlights`; not re-implemented.

**Models the shipped stages run**
- **SAM 2.1 hiera-large** (Meta). https://github.com/facebookresearch/sam2 (Apache-2.0). The box prompts behind every part mask and the disc, and the box and point prompts of the subject check.
- **OWLv2** and **Florence-2** (above).
- **BiRefNet.** Zheng et al. (CAAI AIR 2024). https://arxiv.org/abs/2401.03407 ; weights https://huggingface.co/ZhengPeng7/BiRefNet_dynamic (MIT). The matte behind the on-the-object gates, the backdrop decisions and the subject check.
- **ViTMatte.** Yao et al. (2023). https://arxiv.org/abs/2305.15272 ; code https://github.com/hustvl/ViTMatte (MIT; the small composition-1k weights are Apache-2.0). The edge snap, which never moves a part's pixels.
- **Careaga and Aksoy**, Colorful Diffuse Intrinsic Image Decomposition in the Wild (SIGGRAPH Asia 2024). https://arxiv.org/abs/2409.13690 ; code https://github.com/compphoto/Intrinsic (academic use only). The albedo the grouping clusters and the shading layer the shadow test reads. Its albedo carries the reflections the pruning cannot remove.
