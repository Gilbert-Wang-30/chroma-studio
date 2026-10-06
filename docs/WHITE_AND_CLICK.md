# White paint and click-to-segment: diagnosis, fixes and verified numbers

> **Local-only sources.** Paths under `scratch/` are the author's working files: diagnosis kit, experiment, integration rounds, checks and renders. `scratch/` is git-ignored, so the numbers that cite those files can only be recomputed on this machine. The comparison sheets are copied into `docs/comparison/white/` and `docs/comparison/clickseg/`.

Date 2026-10-01. This covers the owner's request: "when the original colour is white-ish the colour change looks like dogshit, I believe it's falsely considering white as highlights".

**The test set.** Six white photos, each analysed fresh at Balanced:

| Photo | Job | White groups |
|---|---|---|
| Alpine A110, `car_alpine_1` | `68d9a72d7b8e` | body `Off-white` g3, 183,489 px |
| Fila sneakers, `sneakers_2` | `7664f0b665ea` | upper `Light gray` g0, 409,113 px |
| RX-78 RG, `gundam_rx78_rg` | `74229e8c4b12` | armour `Light gray` g2, 373,471 px |
| Unicorn outdoors, `gundam_unicorn_pg_sky` | `8b909a8162d6` | armour split into four groups, g7/g6/g3/g2 |
| Nu EG, `gundam_nu_eg` | `4f96d12cf0df` | armour `Light gray` g1, 239,584 px |
| Airplane model, `airplane_model_1` | `4bea0d465f16` | hull `Gray` g0, 1,032,728 px, under warm light |

**The controls.** Two saturated photos: the red Ducati (`motorcycle_1`, g1 `Red`) and the yellow BMW (`motorcycle_2`, g2 `Amber`).

**Targets.** Navy `#123f9e`, red `#c1121f`, green `#1b4d3e`, pastel `#7fb2ff`, black `#000000` and grey `#7a7a7a`.

**Engines compared.**
- **Frozen** is the engine before this work, kept at `scratch/whiteexp/base`.
- **D5** is the experiment's deliverable.
- **Shipped** is the live `recolor/engine.py` (md5 `20f1f83a…`, equal to `scratch/whiteint/ref/engine_r3.py`).

`.venv/bin/python -m pytest tests/` gives **513 passed** in 24.5 s; I re-ran it for this report. Every number below was reproduced by an independent verifier or checker. Where a verifier corrected a claim, the corrected value is used.

**Metrics.** They come from `scratch/whiteexp/diag/whitekit.py`.
- *Chroma ratio (Cr)*: the render's OK chroma divided by the target's, with the target lit by the photo's shading. Below 1 means washed out.
- *Lift*: log2 of the render's luminance over that same reference.
- *Lit bins*: lit plus bright pixels.
- *Wash*: a damage score summed over 6 photos x 6 targets.
- *Rim*: whitish photo pixels still whitish just inside or outside the label.
- *Bleed*: pixels changed outside the painted groups.
- *Glint white share*: the share of real photo-white glints that are still near-white after the repaint.

## 1. Why white paint repainted badly

**Was the owner right?** In spirit, yes. The engine did treat a white paint's light as if it were a highlight. But the cause is the **residual rule**, not the clipped-highlight test. On white groups the clipped-highlight test never fired: the highlight and gloss masks were 0 px in every white group.

What the decomposition does with white (Careaga v2.1):
- Most of the brightness goes into the shading (median 0.87-0.91, p90 1.2-1.3). The albedo sits at Y 0.61-0.82.
- An achromatic leftover of 2-18 % of the photo stays in the residual. On the sun- and sky-lit Unicorn it is 17-50 %.
- On saturated paint the same leftover is chromatic, and its achromatic floor is 0.00. That is why the frozen rules worked on the bikes and failed on white.

**Shares of the damage.** Each share was measured by switching one rule off at a time on the frozen engine. Line numbers refer to the frozen copy.

| # | Rule and code | What it did to white | Evidence | Share of damage |
|---|---|---|---|---|
| M1 | The achromatic floor of the positive residual was kept as `spec + (1-spec)·follow`, with `spec` normalised by the image's q99 (`engine.py _adjusted_residual` l.1473-1502, `_spec_weight` l.856-870) | All of a white's leftover was read as "glint or veil". It was actually broad diffuse paint energy: only 13-60 % of it sits above a 9 px opening, and r(floor, shading) is +0.70 on the Unicorn. It was re-added as a neutral grey film, 26-63 % of the rendered luminance on lit faces. Result: pastel lit faces, the target colour only in the shadows, blotches, and black rendered as grey marble. | `floor0` (floor not re-added): wash 9.48 → 1.90; navy lit Cr RX-78 0.73 → 1.00, Nu 0.80 → 1.00; black texture r 0.32 → 0.91 (RX-78); controls unchanged. Moving the floor into the shading instead gives −48 %: the damage comes from re-adding it as grey, not from its existence. | ~80 % of the washout, ~99 % of the black-as-charcoal look, ~27 % of the rim |
| M2 | Clipped-highlight test and white gloss floor, both gated by `GLOSS_RATIO_MAX 0.5` (`_gloss` l.958-981, `_highlight` l.872-885) | White groups have a photo min/max of 0.91-0.98, so the gate was always shut. A white's glints sit in the shading (S 1.02-1.18 against 0.90-1.07 in the ring around them), so the repaint painted over them: flat matte plastic. | Glint white share at navy: 0.000-0.014 on every white photo, against 0.77 on the Ducati and 0.49 on the BMW. Forcing the gate open (`gloss_white`) keeps the paint itself white: rim +48 %. | 100 % of the lost highlights |
| M4 | Lightness slope above the anchor, `max(slope_dn, …)` up to 1 (`_group_params` l.1150-1153, `_repaint` l.1422-1440) | Lighting baked into a white albedo was copied 1:1 onto dark targets and pushed light targets past L 1, giving pure-white patches. | `slope_up_dn`: −11 % wash. `L_flat`: pastel near-white lit share Unicorn 71 → 53 %, airplane 56 → 0 %. | ~11-16 % of the washout; nearly all of the pastel white-out |
| M5 | Grouping: `absorb_lit` skips neutral groups (`grouping.py` l.721-731); ViTMatte snaps only groups of chroma ≥ 18 (`matting.py` l.49) | A white lit by varied light stays split into several groups, and white labels keep SAM's short edges. | One click on the Unicorn's g7 repaints 50,274 of 465,920 armour px (10.8 %). The airplane's g0 is 61 % of the hull. | All of the "only part repainted" damage; ~73 % of the rim left after M1 |
| M6 | Reflections (rule 8) whenever hue confidence > 0 (`_group_params` l.1160) | A white's colour cast was taken as the old paint's hue: the sneakers' sky cast (C 2.8) and the airplane hull's warm light (C 10.5). Pixels of that hue nearby were recoloured as its reflections. | `refl_off`: bleed 238,120 → 4,967 px. Airplane 40,074 → 648. Sneakers 1,035 → 13 (the FILA letters and laces). | ~98 % of the bleed |
| M7 | White gloss floor where the light, not the paint, lowers the min/max ratio | The warm-lit hull reads min/max 0.38, so 15.2 % of it kept white "gloss" over the new colour. | `gloss_off`: airplane wash −19 %. It also kills the controls' glints (0.77 → 0.24), so the floor has to stay for saturated paint. | ~3.4 % overall |
| M8 | `materials.absorb_highlights` and the shiny badge | The RX-78's white shield handle (2,864 px) was moved into the red `Rose` group as its "highlight". | `sheets/regions_gundam_rx78_rg.jpg` | Small in area, but visible |
| M9 | Hue turn, chroma scale, bounce retint, gamut, `EXCESS_FOLLOW_GAMMA`, ramp gate | Effectively off for C_A 0.001-0.007. | Each ≤ 3.5 % of the wash | Minor |

## 2. What was tried and what shipped

**Experiment** (`scratch/whiteexp/exp/`, deliverable D5).

Kept:
- A per-group **neutral weight**: CIELAB chroma ramp 10-18, times the larger of an albedo-lightness ramp (L 45-60) and a photo-white-share ramp (0.04-0.10). Groups tagged chrome keep the saturated-paint rules.
- On neutral sources, the residual floor **follows the new paint like the product** (`FLOOR_DIFFUSE_GAMMA 1`), and so does the neutral coloured excess (`NEUTRAL_EXCESS_GAMMA 1`). The latter took wash from 0.78 to 0.50.
- The slope **above the lightness anchor is flat** (`NEUTRAL_UP_K 0`). Pastel white-out went from 1.24 to 0.50.
- An **exposure bound**: where the photo is white, a repaint of albedo A' reflects at most photo x A'/0.8. The light is read within 4 px, and the bound is gated by the group's white share.
- Saturated-paint gloss off for neutral groups.
- Reflections off for neutral sources. Bleed went from 238,120 to 5,036 px.

Rejected:
- Keeping the floor's sharp peaks as glints, in six variants. Each put white speckle along every RX-78 bevel; specks went from 107k up to 181-426k.
- Local-contrast white-glint detectors, in eight variants. They also kept blobs on a sneaker crease, the RX-78 vent rim and a dotted line on the Nu's edge.
- Light scaling k = Y_A/0.8.
- Growing the label 5 px (bleed +22k, cobblestones painted).
- A ViTMatte snap for white groups (rim −0.6 %, and it took 831 px of the red shield).

The M8 grouping guard (`HL_REM_REL 0.35`) is delivered as an option only, in `optional/materials.diff`. It is correct, but its one case looks worse: the handle keeps the shield's pink bounce.

**Verifier on D5.** All 3,585 result leaves reproduced exactly. It raised four issues:
- **No glint stays white anywhere** (36 of 36 renders).
- Fine detail is under-kept: 70-93 % of the photo's high-pass amplitude on chromatic targets, 38-80 % on black.
- With reflections off for neutral sources, white edge pixels inside neighbouring logo groups stay pale. The clearest case is FILA lettering speckle. The pale ring on navy rose from 72,866 to 90,025 px (+24 %).
- The texture gain had been stated as a correlation, which overstates it.

**Integration rounds** (`scratch/whiteint/`, now live). Three rounds addressed those issues. The shipped engine adds the following on top of D5:
- **Rule 7e, white glints.**
  - A small clipped glint is kept whole and white once it passes the old hard cuts. The ramps end at those cuts: area 4-8 px, ring 4-8 px, standout 1.20-1.50x the ring, shading-explained 1.28-1.44x.
  - The lit-face test applies only where the shading steps up under the spot (ramp 1.08-1.20).
  - The whole clipped core takes the photo's brightness.
- **Glint share counted softly** around the analysis stage's cuts (clip 0.97-0.99 sRGB, residual 0.3-0.7 of q99). The gloss-gate ramp is 0.12-0.25, so there is no hard flip.
- **Rule 8 for near-neutral sources** fades in from CIELAB chroma 2 to 8. From chroma 8 the weight is exactly 1, as in the old engine.
- **Deterministic small clips**: the per-spot sums are float64 on the CPU. Four builds give bit-identical fields.
- Every ramp is smooth. The white-paint rules fade out as the target nears the source colour; all map-to-own-colour cases are identical to the old engine.

No job needs re-analysing; the change is render-time only. README rule 10 and `docs/ARCHITECTURE.md` §3.5 are updated. **The engine-invariants memory file has not been edited and needs your decision.** Suggested tenth invariant: "a white paint's residual leftover is its diffuse light, not a glint; the white-paint rules apply only to neutral sources, fade out near the source colour, use smooth ramps, and keep a small clipped glint whole once it passes the old cuts."

## 3. Verified numbers, before and after

**Totals** (6 white photos x 6 targets; lower is better unless noted)

| Measure | Frozen | D5 | Shipped |
|---|---|---|---|
| Wash | 9.48 | 0.50 | **0.45** |
| Black lift | 12.89 | 0.46 | 0.47 |
| White rim (px) | 65,778 | 40,503 | 43,099 |
| Bleed (px) | 238,120 | 5,442 | 6,611 |
| 1 − texture r | 4.91 | 1.92 | 2.45 (1.966 outside kept glints) |
| Specks (px) | 388,501 | 15,285 | 18,801 (14,877 outside kept glints) |
| Blotch | 13.07 | 8.33 | 8.48 |
| Pastel lit near-white | 1.27 | 0.04 | 0.06 |
| Pale 6 px ring outside the groups, navy (px) | 72,866 | 90,025 | 88,752 |

**Shipped wash per photo:** Alpine 0.05, sneakers 0.02, RX-78 0.05, Unicorn 0.26, Nu 0.02, airplane 0.05.

**Lit bins, frozen → D5.** The shipped wash is equal to D5's or lower on every photo.

| Photo | Navy Cr | Navy lift | Black lit OK L excess |
|---|---|---|---|
| Alpine | 0.92 → 1.00 | +0.29 → +0.04 | +0.06 → ≤ 0.05 |
| Sneakers | 0.81 → 0.96 | +0.64 → −0.03 | +0.09 → ≤ 0.05 |
| RX-78 | 0.73 → 1.01 | +0.80 → +0.06 | +0.18 → ≤ 0.05 |
| Unicorn | 0.83 → 0.98 | +1.84 → +0.06 | +0.29 → ≤ 0.05 |
| Nu | 0.80 → 1.01 | +0.56 → −0.01 | +0.14 → ≤ 0.05 |
| Airplane | 0.74 → 0.95 | +2.48 → +0.49 | +0.44 → ≤ 0.05 |

- Green Cr went from 0.28-0.70 to 0.94-1.05.
- Black texture r went from 0.32-0.79 to 0.76-0.98.
- Unicorn one-click case (g7 alone): Cr 0.81 → 1.02, lift +0.62 → +0.12, texture r 0.78 → 0.96.

**Real-glint white share, shipped** (navy, red, green and black):

| Photo | D5 | Shipped |
|---|---|---|
| Airplane | 0 | 0.93 |
| Alpine | 0 | 0.46 |
| Unicorn | 0 | 0.43-0.45 |
| Nu | 0 | 0.09 |
| RX-78 | 0 | 0.00 |

- The Alpine roof streak renders at median OK L 1.00 on navy (photo 0.99) and 0.99 on black.
- Local-highlight log contrast on navy, frozen / D5 / shipped: sneakers 0.68 / 0.48 / 0.558, Unicorn 0.93 / 0.65 / 0.687.

**Saturated sources and controls.**
- Ducati at navy: Cr 1.08, lift +0.39, glint 0.77, bleed 5,705.
- BMW at navy: Cr 1.07, lift +0.75, glint 0.49, bleed 515.
- Both are identical across frozen, D5 and shipped.
- Bit for bit: all 162 groups of CIELAB C ≥ 18 in the 8 jobs, each mapped alone to navy, black and red; the empty mapping on all 8 jobs; and the halo harness (halo 5,573, bleed 206,066, identity 0, largest pixel difference 0).
- **Exception:** a saturated group *co-mapped with a white group* differs from the frozen engine on its own pixels. The largest cases are 72 levels on the Nu's `Navy` (C 18.1, 3,032 px), 53 on the Unicorn's `Navy` and 34 on the RX-78's `Rose`. This dates from the earlier rounds. "Coloured paints render as before" is only true for a saturated source mapped without a white-paint group.

**Continuity.**
- Source sweeps (L 60-98 and C 0-15 at 0.25 steps, two hues, three targets): without a glint the largest step is 6 levels. With a glint it is 17 levels, where the glint ramp passes, against 99-237 in round 1.
- Shading step under a glint: 24 levels per 0.01 at most, 28 with a bright halo. This axis has no unit test.

**Time.**
- Steady renders are unchanged: preview 5.8-8.5 ms, working size 8.4-17.1 ms.
- The first preview of a new renderer on a white photo is slower: 90-119 ms, against 47-75 ms before.
- Exports run end to end in 0.44-4.6 s on an idle GPU. A loaded host measured 6.0-6.4 s for the Alpine at full resolution.

## 4. Judges' verdict

Three independent judges read the four white sheets in `docs/comparison/white/`. Each sheet shows ORIGINAL | CURRENT | NEW across navy, red, pastel and black, with 3x close-ups. CURRENT is the frozen engine and NEW is the working-tree engine; both ran on fresh analyses with the same group ids. **All three ranked NEW above CURRENT.**

| Target | CURRENT (judges 1 / 2 / 3) | NEW (judges 1 / 2 / 3) |
|---|---|---|
| Navy | 4 / 4.5 / 4 | 7 / 7 / 7.5 |
| Red | 3.5 / 4 / 4 | 7.5 / 7 / 7.5 |
| Pastel | 5 / 6 / 5.5 | 7.5 / 7 / 7.5 |
| Black | 2.5 / 2.5 / 2 | 5 / 4.5 / 5.5 |

**What CURRENT got wrong.** All three described it as a translucent tint over white: periwinkle navy, salmon-pink red, a blown-white pastel on the Unicorn, and grey marble instead of black, with white hairlines along panel gaps and stitching.

**What NEW fixes.** Solid target colour on every lit face, and the Unicorn and RX-78 glint stripes stay white.

**What all three faulted in NEW:**
- **Black** flattens into a matte silhouette. The Alpine has no reflections, the RX-78 merges into the backdrop, and sneaker stitching and tread vanish.
- **Too little surface life on every target.** The Alpine's hood reflections, louvres and shut lines are gone, and large faces look evenly filled and slightly plasticky.
- **Decals erased**: the Unicorn's `1404` and its Federation markings, and the sneaker emboss.
- **Edges**: hard dark outlines on small parts, a light dotted fringe on black, and stray white chips on the Unicorn's shoulders on black.

Two smaller points:
- Isolated kept glints can read as stark white dashes on black.
- The Alpine and sneaker glint close-ups contain no real glint, so the sheets cannot show glint handling on those two photos.

## 5. Click-to-segment

**How to use it.**
- **Open the tool.** Use **Select part** in the toolbar, press `S`, or Alt-click on the image.
- **Place points.** Click to add a point. Shift-click or right-click leaves an area out. Drag to draw a box.
- **Adjust.** `N` cycles SAM's shapes, `Backspace` removes the last point or the box, and `Esc` cancels.
- **Commit.** Press `Enter` to make a group, after typing a name or keeping the default. Enter pressed before the outline arrives waits for it.
- **Read the pill before Enter.** It says what Enter will do:
  - "takes in Crimson": a colour group is absorbed whole.
  - "replaces Brake caliper": a group is replaced, and its paint and name carry over.
  - "cuts Shock spring (32 % stays)": shown with a **Take all** button, which grows the part over the whole group.
- **Find.** Type a phrase; vocabulary words and synonyms count, so "muffler" finds Exhaust. Each chip is one of three kinds:
  - a check for the existing group;
  - "overlaps", for a candidate that partly covers an existing group;
  - an alert for a part of another kind.
  Choosing a chip loads it into the pill.
- **Remove.** Remove on a drawn part gives its pixels back, along with any groups it took in whole and their paint.
- **Regroups.** Parts survive Auto-regroup with identical pixels, name, paint and flags.

Screens: `docs/comparison/clickseg/clickseg_flow.jpg`, with full frames 01-08 alongside. The flow covers:
- a Ducati far caliper: one click, then two more points and a Shift-click on the disc, then Enter, then painted red;
- a Corvette: Find "brake caliper", then candidate 3, which replaces the detected caliper.

**Latency** (as verified, on a shared GPU):

| Step | Time |
|---|---|
| Prompt, 1-2 clicks, median at the API | 32.7 ms (max 41.5) |
| Prompt, 3 clicks, median | 35 ms |
| First prompt on a new job | 113 ms |
| First prompt after the 60 s idle unload | 1,094 ms |
| Click to outline in the browser, median of 17 clicks | 119 ms (all ≤ 150 ms) |
| Warm commit | 139-167 ms (medians 148 / 157 ms, host load average 17); the claimed 0.11-0.12 s did not reproduce |
| Remove | 91-117 ms |
| Find | 0.56-1.59 s (OWLv2 0.28-0.57 s) |
| Render after painting a part | 19-22 ms |

**Robustness, verified.**
- **Malformed input**: 125 of 125, 29 of 29, plus 34 JSON and 11 raw bodies on `/segment` all got clean 4xx. NaN, Infinity, 1e400 and 10\*\*400 are 400s. Chunked bodies over 1 MB are 413s.
- **Concurrency**: races of 4 commits plus a regroup left the partition intact, with no hung thread.
- **GPU contention**: prompts and commits waiting for the GPU answer 503 with Retry-After within about 1.9 s.
- **Fuzzing**: 200 seeds found no failure of the partition or registry invariants.
- **Fixtures**: the six fixture jobs were untouched.

**Known defects.** The UI check failed on one major issue. The code checks and the verifier found the rest, all minor.
- **Major: stale accessible names in the Mapping panel** (`web/js/panels/mapping.js`). After an edit renumbers groups (Remove, replace or merge), a reused row shows the new name, but its picker and button `aria-label`s still name the old group. A screen reader, or anyone picking by label, paints the wrong group. In the UI check, `#ffcc00` meant for Shock spring landed on `Copper`. Fix: refresh both labels in `updateRow()`.
- **Remove is not always exact.** A cut of under 12 px into a covered region is not recorded, so 1-5 px drift between groups for good. This happens on a repeated commit of the same selection, and in 7 of 200 fuzz seeds.
- **Remove after merges and splits.** Remove leaves a merged detected region as a new group. A colour split of a user part followed by Remove on one half leaves an orphan with its carve record deleted.
- **Locks.** A replaced, hand-locked group is re-locked by the next Auto-regroup.
- **Box prompts are not held to the box.** A 6 px edge box committed 871,586 px, the whole backdrop.
- **Detected part instances.** Selecting one instance of a detected part (one of two mirrors) drops its name and paint.
- **Share mismatch.** The browser measures the "cuts X (N % stays)" share over the whole group, while the server absorbs per instance. With multi-instance groups the note is wrong, and Take all pulls in the far instance.
- **Enter before the outline** skips the "cuts" warning. One caliper was committed half painted that way.
- **Find on old jobs** without part groups ranks the exhaust first for "spring".
- **UI layout.**
  - On a drawn part's row the trash button squashes the size bar and misaligns the percentage.
  - At 1024 px the status-bar hints are cut off.
  - Choosing a Find chip makes the list jump under the pointer.
  - The selected row scrolls out of view after a regroup.
- Keyboard users cannot add points.
- The mock server fakes `/segment` and returns 501 for `groups/from_mask`.

## 6. Remaining gaps

**White paint**
1. **Black on white paint is a matte silhouette.** Broad unclipped reflections, shut lines, louvres and decals are flattened on dark targets. This was the judges' main complaint, and the next thing to fix.
2. **Glints are kept only where they clip.** The RX-78 keeps 0.00 of its real glints and the Nu 0.09. Spots just below the cuts stay as faint dots: the airplane's dimmer lamp dots at weight about 0.5, a dotted pale edge line on the RX-78 on black, and pale dashes on Nu edges. Kept glints are hard-edged with almost no falloff; the airplane's lamp band becomes about 8 dots.
3. **Rim and fringe.** A 1-5 px white rim remains (43,099 px). The pale ring outside the groups is about 88.8k px on navy, against 72.9k frozen; the FILA speckle is the clearest case. Labels stop short and white groups have no matte snap.
4. **Grouping.**
   - The Unicorn's armour is four lightness groups, so one click paints 10.8 % of it.
   - The airplane hull is three groups, and the analysis flags it as backdrop: you have to turn off "Ignore background" to paint it. Its navy lit faces are still +0.49 over target.
   - The RX-78's shield handle sits in the red group. The optional guard fixes this, but it needs a rule-8 bounce fix first.
5. **Partial white-paint rules on greys.** Light-grey non-chrome parts (albedo L ≥ 60) and the Ducati's `Silver` (red top 10 % at 0.53, against 0.62 before) get them in part. The exposure bound acts only where the photo is white, and assumes white reflects about 80 %.
6. **Co-mapped saturated groups** differ from the frozen engine by up to 72 levels (section 3).
7. **Tests.** Two soft ramps have no test that catches their reversion: the small-clip trust and the lit-count ramp. The shading-step axis has no sweep. `_client_gone` has no automated test.

**Housekeeping**
- The memory file needs your decision on the new invariant (section 2).
- Scratch that can be deleted: `scratch/whiteint` (1.2 GB), `scratch/whiteexp/exp/jobs_x` (527 MB, can be regenerated) and the `scratch/whiteexp/checks` and `scratch/clickseg/checks` outputs.
- The experiment agent used `pkill` once on a pattern matching only its own sweep. The app on 8810 was unaffected, and every later stop used explicit PIDs.
