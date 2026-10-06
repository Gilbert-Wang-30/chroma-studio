"""Part-aware grouping (recolor/segmentation/grouping.py, refine.py, materials.py) on synthetic
label maps: a detected part (``part_kind`` in the regions stage's info) is one group per
kind whatever its colour, is held out of every colour step, splits into its instances and
joins them again by name; the Groups panel's view (minor rows and their parents). No models:
the boundary snap is a stand-in."""
from __future__ import annotations

import numpy as np
import pytest

from recolor import imageio
from recolor.segmentation import grouping, refine
from recolor.types import ColorGroup, Region

YELLOW = (80.0, 5.0, 75.0)
GOLD = (55.0, 8.0, 45.0)
GREY = (60.0, 0.0, 0.0)
RED = (45.0, 60.0, 45.0)


def _albedo(labels, colours):
    lab = np.zeros(labels.shape + (3,), np.float32)
    for rid, col in colours.items():
        lab[labels == rid] = col
    return np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)


def _photo(albedo, shade=0.9):
    return imageio.to_uint8(imageio.linear_to_srgb(albedo * shade))


def _part(rid, kind="shock_spring", label="Shock spring", plural="Shock springs", instance=0, **kw):
    return {"id": rid, "source": "kind", "part_kind": kind, "part_label": label, "part_plural": plural,
            "part_instance": instance, "confidence": 0.9, **kw}


def _scene():
    """0: grey backdrop, 1: the yellow frame, 2: a yellow spring on it (a detected part), 3: the
    other spring (a second instance), 4: a gold part."""
    h, w = 80, 120
    labels = np.zeros((h, w), np.int32)
    labels[10:70, 10:110] = 1
    labels[20:40, 20:30] = 2
    labels[20:40, 90:100] = 3
    labels[50:60, 50:70] = 4
    albedo = _albedo(labels, {0: GREY, 1: YELLOW, 2: YELLOW, 3: YELLOW, 4: GOLD})
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, _part(2), _part(3, instance=1),
            {"id": 4, "source": "sam"}]
    return labels, albedo, info


def _group_of(regions, groups):
    return {r.id: next(g for g in groups if g.id == r.group_id) for r in regions}


# ------------------------------------------------------------------ the clustering

def test_a_part_kind_is_its_own_group_whatever_its_colour():
    labels, albedo, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info, photo_rgb_u8=_photo(albedo))
    g = _group_of(regions, groups)
    assert g[2].id == g[3].id != g[1].id                                 # both springs, apart from the frame
    springs = g[2]
    assert springs.part == "shock_spring" and springs.part_instances == 2 and springs.name == "Shock springs"
    assert not springs.is_background and not springs.locked
    assert g[1].part == "" and g[0].is_background
    # the same regions untagged: the springs are the frame's colour and share its group
    plain = [dict(d, source="sam", part_kind="") for d in info]
    regions, groups, _ = grouping.group_regions(labels, albedo, plain, photo_rgb_u8=_photo(albedo))
    g = _group_of(regions, groups)
    assert g[2].id == g[1].id and not any(x.part for x in groups)


def test_two_kinds_of_one_colour_and_a_tiny_part_keep_their_own_groups():
    labels, albedo, info = _scene()
    info[3] = _part(3, kind="grip", label="Grip", plural="Grips")
    labels[60:62, 20:24] = 5                                             # an 8 px footpeg
    albedo = _albedo(labels, {0: GREY, 1: YELLOW, 2: YELLOW, 3: YELLOW, 4: GOLD, 5: YELLOW})
    info.append(_part(5, kind="footpeg", label="Footpeg", plural="Footpegs"))
    regions, groups, _ = grouping.group_regions(labels, albedo, info, photo_rgb_u8=_photo(albedo))
    g = _group_of(regions, groups)
    assert len({g[1].id, g[2].id, g[3].id, g[5].id}) == 4
    assert (g[2].name, g[3].name, g[5].name) == ("Shock spring", "Grip", "Footpeg")


def test_a_part_is_never_background_and_stays_out_of_the_group_cap():
    labels, albedo, info = _scene()
    info[2]["bg"] = 2                                                    # the matte called the spring backdrop
    regions, groups, _ = grouping.group_regions(labels, albedo, info, max_groups=1)
    g = _group_of(regions, groups)
    assert not next(r for r in regions if r.id == 2).backdrop
    assert g[2].part and not g[2].is_background
    colour = [x for x in groups if not x.part]
    assert len(colour) == 1 and not colour[0].is_background              # a lone colour group stays paintable
    assert len(groups) == 2                                              # one colour group + the spring kind


def test_the_lit_merge_never_touches_a_part():
    """The lit and the shadowed side of one paint merge; the shadowed side tagged as a part does not."""
    h, w = 80, 140
    labels = np.zeros((h, w), np.int32)                                  # 0: a grey backdrop
    labels[10:70, 10:90] = 1                                             # 1: the lit side of a red paint
    labels[10:70, 90:130] = 2                                            # 2: its shadowed side
    lit, dark = (55.0, 60.0, 40.0), (35.0, 42.0, 28.0)
    albedo = _albedo(labels, {0: GREY, 1: lit, 2: dark})
    photo = imageio.to_uint8(imageio.lab_to_rgb(imageio.linear_to_lab(albedo)))
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, {"id": 2, "source": "sam"}]
    regions, groups, _ = grouping.group_regions(labels, albedo, info)
    assert len(groups) == 3                                              # the plain linkage keeps the sides apart
    regions, groups, _ = grouping.group_regions(labels, albedo, info, photo_rgb_u8=photo)
    assert len(groups) == 2                                              # one paint under two lights
    info[2] = _part(2, kind="fender", label="Fender", plural="Fenders")
    regions, groups, _ = grouping.group_regions(labels, albedo, info, photo_rgb_u8=photo)
    assert len(groups) == 3 and any(g.part == "fender" for g in groups)


def test_a_plain_regroup_keeps_the_part_groups():
    labels, albedo, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    r2, g2, _ = grouping.regroup(regions, labels, albedo, max_groups=None, delta_e=40.0)
    g = _group_of(r2, g2)
    assert g[2].part == "shock_spring" and g[2].id == g[3].id != g[1].id


# ------------------------------------------------------------------ the refinement

def _snap_spy(seen):
    def snap(image, labels, group_map, groups, protect=None, progress=None):
        seen["protect"] = None if protect is None else protect.copy()
        return labels.copy(), "spy"
    return snap


def test_refinement_keeps_a_part_out_of_the_paint_and_its_pixels_out_of_the_snap():
    """A red paint with a dull dark-red detected part: untagged, the material lock locks it;
    tagged, it is a paintable part group, never the paint, and the snap never moves its pixels."""
    h, w = 80, 120
    labels = np.full((h, w), 0, np.int32)
    labels[10:70, 10:110] = 1
    labels[30:50, 40:60] = 2
    albedo = _albedo(labels, {0: GREY, 1: RED, 2: (30.0, 20.0, 15.0)})
    photo = _photo(albedo)
    base = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, {"id": 2, "source": "sam"}]
    regions, groups, gm = grouping.group_regions(labels, albedo, base, photo_rgb_u8=photo)
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm, _snap_spy({}))
    assert _group_of(res.regions, res.groups)[2].locked                  # another material, locked
    tagged = base[:2] + [_part(2, kind="brake_caliper", label="Brake caliper", plural="Brake calipers")]
    regions, groups, gm = grouping.group_regions(labels, albedo, tagged, photo_rgb_u8=photo)
    seen: dict = {}
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm, _snap_spy(seen))
    g = _group_of(res.regions, res.groups)
    assert g[2].part == "brake_caliper" and not g[2].locked and g[2].name == "Brake caliper"
    assert refine.main_paint(res.groups).id == g[1].id
    assert seen["protect"][labels == 2].all()                            # the part's pixels are kept from the snap
    assert res.report["parts"] == [{"id": g[2].id, "name": "Brake caliper", "kind": "brake_caliper", "instances": 1,
                                    "area": 400}]


def test_a_part_is_never_the_paint_even_when_it_is_the_largest_colour():
    groups = [ColorGroup(id=0, name="Wheel rims", albedo_lab=RED, albedo_hex="#c03020", area=900, area_frac=0.5,
                         region_ids=[0], hue_family="red", part="rim", part_label="Wheel rim", part_plural="Wheel rims",
                         part_instances=2),
              ColorGroup(id=1, name="Red", albedo_lab=RED, albedo_hex="#c03020", area=500, area_frac=0.3,
                         region_ids=[1], hue_family="red")]
    assert refine.main_paint(groups).id == 1 and refine.paint_family(groups) == [1]
    assert refine._nearest_group(RED, groups) == 1                       # a decal never joins a part by colour


def test_the_washed_absorb_leaves_parts_alone():
    labels = np.zeros((60, 100), np.int32)                               # 0: a grey backdrop around them
    labels[10:50, 10:50] = 1                                             # 1: red paint
    labels[10:50, 50:90] = 2                                             # 2: the paint washed out by a highlight
    lab = np.zeros((60, 100, 3), np.float32)
    lab[...] = GREY
    lab[labels == 1] = (45.0, 60.0, 45.0)
    rng = np.random.default_rng(0)
    L = rng.uniform(50.0, 62.0, 40 * 40).astype(np.float32)
    C = 50.0 - 1.5 * (L - 50.0)
    hue = np.arctan2(45.0, 60.0)
    lab[labels == 2] = np.stack([L, C * np.cos(hue), C * np.sin(hue)], 1)
    albedo = np.clip(imageio.lab_to_linear(lab), 0, 1).astype(np.float32)
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, {"id": 2, "source": "sam"}]
    regions, groups, _ = grouping.group_regions(labels, albedo, info)
    _, _, _, moves = refine.absorb_washed(regions, groups, labels, imageio.linear_to_lab(albedo))
    assert [m["region"] for m in moves] == [2]                           # washed-out paint joins the paint
    info[2] = _part(2, kind="tank", label="Fuel tank", plural="Fuel tanks")
    regions, groups, _ = grouping.group_regions(labels, albedo, info)
    _, _, _, moves = refine.absorb_washed(regions, groups, labels, imageio.linear_to_lab(albedo))
    assert moves == []                                                   # a detected part never moves


def test_regroup_refined_reproduces_the_part_groups_from_the_seed():
    labels, albedo, info = _scene()
    photo = _photo(albedo)
    regions, groups, gm = grouping.group_regions(labels, albedo, info, photo_rgb_u8=photo)
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm, _snap_spy({}))
    tags = {d["id"]: {"kind": d["part_kind"], "label": d["part_label"], "plural": d["part_plural"],
                      "instance": d["part_instance"]} for d in info if d.get("part_kind")}
    bg = np.array([2, 0, 0, 0, 0], np.int8)
    r2, g2, gm2, _ = refine.regroup_refined(photo, albedo, labels, res.origin, res.labels, res.regions, res.islands,
                                            None, 10.0, bg=bg, part_tags=tags)
    before = sorted((tuple(g.region_ids), g.name, g.part) for g in res.groups)
    assert sorted((tuple(g.region_ids), g.name, g.part) for g in g2) == before
    r3, g3, _, _ = refine.regroup_refined(photo, albedo, labels, res.origin, res.labels, res.regions, res.islands,
                                          None, 40.0, bg=bg, part_tags=tags)
    assert any(g.part == "shock_spring" and set(g.region_ids) == {2, 3} for g in g3)


def test_enforce_parts_brings_a_stray_part_region_home():
    labels, albedo, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    frame = next(g for g in groups if 1 in g.region_ids)
    regions, groups, gm = grouping.move_regions(groups, regions, gm, labels, [3], frame.id)
    assert _group_of(regions, groups)[3].id == frame.id
    regions, groups, gm = grouping.enforce_parts(regions, groups, labels)
    g = _group_of(regions, groups)
    assert g[3].id == g[2].id and g[2].part == "shock_spring"
    assert np.array_equal(np.array([r.group_id for r in regions])[labels], gm)


# ------------------------------------------------------------------ instances and names

def test_split_instances_names_them_by_position_and_merge_joins_them_again():
    labels, albedo, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    springs = next(g for g in groups if g.part)
    r2, g2, gm2 = grouping.split_instances(groups, regions, gm, labels, springs.id)
    grouping._check_state(r2, g2)
    names = {g.name: g for g in g2 if g.part}
    assert set(names) == {"Shock spring (left)", "Shock spring (right)"}
    assert 2 in names["Shock spring (left)"].region_ids and 3 in names["Shock spring (right)"].region_ids
    assert all(not g.is_background and g.part_instances == 1 for g in names.values())
    assert [g.region_ids for g in g2 if not g.part] == [g.region_ids for g in groups if not g.part]
    r3, g3, _ = grouping.merge_groups(g2, r2, gm2, labels, [g.id for g in names.values()])
    merged = next(g for g in g3 if g.part)
    assert merged.name == "Shock springs" and merged.part_instances == 2 and sorted(merged.region_ids) == [2, 3]
    # a group of one instance is left as it is; the position names survive an unrelated move
    r4, g4, _ = grouping.split_instances(g2, r2, gm2, labels, names["Shock spring (left)"].id)
    assert [g.name for g in g4] == [g.name for g in g2]
    frame = next(g for g in g2 if 1 in g.region_ids)
    r5, g5, _ = grouping.move_regions(g2, r2, gm2, labels, [4], frame.id)
    assert {g.name for g in g5 if g.part} == {"Shock spring (left)", "Shock spring (right)"}


def test_instance_names_for_stacked_and_many_instances():
    assert grouping.instance_names("Mirror", [(10, 50), (12, 5)]) == ["Mirror (lower)", "Mirror (upper)"]
    assert grouping.instance_names("Exhaust", [(30, 0), (10, 0), (20, 0)]) == ["Exhaust (3)", "Exhaust (1)", "Exhaust (2)"]
    assert grouping.instance_names("Seat", [(1, 1)]) == ["Seat"]


def test_the_wheels_of_a_vehicle_split_into_front_and_rear():
    """In a side view left and right are the vehicle's front and rear: the wheels, their calipers
    and the doors are named by the end they sit at when the parts give the front away (the
    sprocket and the exhaust behind the wheels' midpoint, the grips ahead of it), numbered when
    nothing does; twin exhaust tips side by side, and every other kind, keep left / right."""
    h, w = 100, 300
    labels = np.zeros((h, w), np.int32)
    labels[40:90, 20:80] = 1                                         # front tyre (left)
    labels[40:90, 220:280] = 2                                       # rear tyre (right)
    labels[60:75, 235:265] = 3                                       # the sprocket in the rear wheel
    labels[30:36, 190:200] = 4                                       # twin exhaust tips, side by side
    labels[30:36, 206:216] = 5                                       # behind the wheels' midpoint
    albedo = _albedo(labels, {0: GREY, 1: (20.0, 0.0, 0.0), 2: (20.0, 0.0, 0.0), 3: GREY, 4: GOLD, 5: GOLD})
    info = [{"id": 0, "source": "sam"}, _part(1, "tyre", "Tyre", "Tyres", 0), _part(2, "tyre", "Tyre", "Tyres", 1),
            _part(3, "sprocket", "Sprocket", "Sprockets"), _part(4, "exhaust", "Exhaust", "Exhausts", 0),
            _part(5, "exhaust", "Exhaust", "Exhausts", 1)]
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    assert grouping.vehicle_front(regions) == -1                     # the front is on the left
    tyres = next(g for g in groups if g.part == "tyre")
    r2, g2, _ = grouping.split_instances(groups, regions, gm, labels, tyres.id)
    names = {g.name: g.region_ids for g in g2 if g.part == "tyre"}
    assert names == {"Tyre (front)": [1], "Tyre (rear)": [2]}
    exhausts = next(g for g in groups if g.part == "exhaust")
    r3, g3, _ = grouping.split_instances(groups, regions, gm, labels, exhausts.id)
    assert {g.name for g in g3 if g.part == "exhaust"} == {"Exhaust (left)", "Exhaust (right)"}   # 16 px apart
    labels[labels == 4] = 0
    labels[30:36, 100:110] = 4                                       # a header ahead of the muffler: front / rear
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    exhausts = next(g for g in groups if g.part == "exhaust")
    _, g4, _ = grouping.split_instances(groups, regions, gm, labels, exhausts.id)
    assert {g.name: g.region_ids for g in g4 if g.part == "exhaust"} == {"Exhaust (front)": [4], "Exhaust (rear)": [5]}
    assert grouping.instance_names("Tyre", [(250, 60), (50, 60)], front=0, axle=True) == ["Tyre (2)", "Tyre (1)"]
    assert grouping.instance_names("Tyre", [(250, 60), (50, 60)], front=1, axle=True) == ["Tyre (front)", "Tyre (rear)"]
    no_cues = [r for r in regions if r.part_kind == "tyre"]
    assert grouping.vehicle_front(no_cues) == 0


def test_part_names_are_automatic_and_a_rename_is_kept():
    labels, albedo, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    springs = next(g for g in groups if g.part)
    assert grouping._keep_name(springs) is None                          # "Shock springs" is automatic
    springs.name = "Rear shocks"
    assert grouping._keep_name(springs) == "Rear shocks"                 # a custom name survives a rebuild
    carry = {g.id: {"name": grouping._keep_name(g), "locked": g.locked, "is_background": g.is_background} for g in groups}
    _, rebuilt, _ = grouping._finalize(regions, labels, {r.id: r.group_id for r in regions}, carry)
    assert "Rear shocks" in {g.name for g in rebuilt}
    split = ColorGroup(id=0, name="Shock spring (left)", albedo_lab=YELLOW, albedo_hex="#f0d020", area=1, area_frac=0.1,
                       region_ids=[2], hue_family="yellow", part="shock_spring", part_label="Shock spring",
                       part_plural="Shock springs", part_instances=1)
    assert grouping._keep_name(split) == "Shock spring (left)"           # kept until a merge joins the instances
    assert grouping._is_auto_part_name(split, "Shock spring (left)") and grouping._is_auto_part_name(split, "Shock springs 2")


def test_a_group_is_a_part_group_by_area_majority():
    def reg(rid, area, kind=""):
        return Region(id=rid, area=area, bbox=(0, 0, 1, 1), albedo_lab=YELLOW, albedo_hex="#f0d020", group_id=0,
                      touches_border=False, source="kind" if kind else "sam", part_kind=kind,
                      part_label="Seat" if kind else "", part_plural="Seats" if kind else "",
                      part_instance=0 if kind else -1)
    g = grouping._group_from(0, [reg(0, 300, "seat"), reg(1, 100)], 1000)
    assert g.part == "seat" and g.name == "Seat"
    g = grouping._group_from(0, [reg(0, 100, "seat"), reg(1, 300)], 1000)
    assert g.part == "" and g.name != "Seat"


# ------------------------------------------------------------------ the panel view

def test_annotate_marks_tiny_leftovers_minor_next_to_their_parent():
    h, w = 100, 100
    labels = np.zeros((h, w), np.int32)
    labels[10:90, 10:90] = 1                                             # the paint
    labels[40:43, 40:43] = 2                                             # a 9 px sliver in it
    labels[60:63, 20:23] = 3                                             # 9 px of lettering
    labels[70:73, 70:73] = 4                                             # a 9 px detected part
    albedo = _albedo(labels, {0: GREY, 1: RED, 2: (30.0, 40.0, 30.0), 3: (95.0, 0.0, 0.0), 4: YELLOW})
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, {"id": 2, "source": "sam"},
            {"id": 3, "source": "text"}, _part(4, kind="badge", label="Badge", plural="Badges")]
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    grouping.annotate_groups(groups, regions, gm)
    g = _group_of(regions, groups)
    assert g[2].minor and g[2].parent == g[1].id
    assert not g[3].minor and not g[4].minor and not g[1].minor and not g[0].minor
    assert g[1].parent == -1
    back = ColorGroup.from_dict(g[2].to_dict())                          # the job record carries the view
    assert back.minor and back.parent == g[1].id


def test_old_records_load_with_no_part_and_no_panel_view():
    old = {"id": 0, "name": "Red", "albedo_lab": [45.0, 60.0, 45.0], "albedo_hex": "#c03020", "area": 10,
           "area_frac": 0.1, "region_ids": [0], "hue_family": "red"}
    g = ColorGroup.from_dict(old)
    assert g.part == "" and g.part_instances == 0 and not g.minor and g.parent == -1
    r = Region.from_dict({"id": 0, "area": 10, "bbox": [0, 0, 1, 1], "albedo_lab": [45.0, 60.0, 45.0],
                          "albedo_hex": "#c03020", "group_id": 0, "touches_border": False, "source": "sam"})
    assert r.part_kind == "" and r.part_instance == -1


@pytest.mark.parametrize("n_inst", [1, 3])
def test_split_instances_rejects_nothing_and_changes_nothing_on_a_colour_group(n_inst):
    labels, albedo, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    frame = next(g for g in groups if 1 in g.region_ids)
    r2, g2, _ = grouping.split_instances(groups, regions, gm, labels, frame.id)
    assert [(g.region_ids, g.name) for g in g2] == [(g.region_ids, g.name) for g in groups]


def test_a_tiny_group_of_a_colour_of_its_own_is_no_minor_group():
    """Minor rows are lighting variants of their neighbour; a tiny group more than dE 15 from
    every neighbour (a gold preload adjuster next to silver) is a real small part and stays
    among the colours."""
    h, w = 100, 100
    labels = np.zeros((h, w), np.int32)
    labels[10:90, 10:90] = 1                                             # the paint
    labels[40:43, 40:43] = 2                                             # 9 px of its shadow
    labels[60:64, 60:64] = 3                                             # 16 px of a gold part
    albedo = _albedo(labels, {0: GREY, 1: RED, 2: (30.0, 40.0, 30.0), 3: (70.0, 5.0, 60.0)})
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, {"id": 2, "source": "sam"},
            {"id": 3, "source": "sam"}]
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    grouping.annotate_groups(groups, regions, gm)
    g = _group_of(regions, groups)
    gold, paint = g[3], g[1]
    assert float(imageio.delta_e(np.array([gold.albedo_lab], np.float32), np.array([paint.albedo_lab], np.float32))[0]) > 15
    assert not gold.minor and gold.parent == -1                          # a colour of its own: a Colours row
    assert g[2].minor and g[2].parent == paint.id                        # the shadow: minor, next to the paint


def test_a_lone_colour_group_is_never_background_after_any_rebuild():
    """max_groups 1 with a detected part: the one colour group holds the whole object and the
    backdrop; flagged background it would leave nothing paintable. Every rebuild (the washed
    absorb, the part enforcement, the pruning) goes through _mark_background, which keeps it
    unflagged, and the part group is never background either."""
    labels = np.zeros((200, 200), np.int32)                              # a big backdrop (0) ...
    labels[80:120, 80:120] = 1                                           # ... a small object (1) ...
    labels[90:100, 90:100] = 2                                           # ... with a detected part on it
    albedo = _albedo(labels, {0: GREY, 1: YELLOW, 2: YELLOW})
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, _part(2)]
    regions, groups, gm = grouping.group_regions(labels, albedo, info, max_groups=1)
    colour = [g for g in groups if not g.part]
    assert len(colour) == 1 and len(groups) == 2                         # mostly backdrop, yet not background
    assert all(not g.is_background for g in groups)
    lab = imageio.linear_to_lab(albedo)
    r2, g2, _, _ = refine.absorb_washed(list(regions), list(groups), labels, lab)   # a rebuild
    assert all(not g.is_background for g in g2)
    r3, g3, _ = grouping.enforce_parts(r2, g2, labels)
    assert all(not g.is_background for g in g3)
    # with a second colour group the backdrop is flagged again, the part never
    labels[20:40, 20:40] = 3
    albedo = _albedo(labels, {0: GREY, 1: YELLOW, 2: YELLOW, 3: RED})
    info.append({"id": 3, "source": "sam"})
    regions, groups, gm = grouping.group_regions(labels, albedo, info, max_groups=2)
    assert any(g.is_background for g in groups) and not any(g.is_background for g in groups if g.part)


def test_a_colour_split_of_a_part_group_names_the_pieces_by_position():
    """A colour split that separates the instances of a part group names them as a split by
    instance would ("Shock spring (left)" / "(right)"), not "Shock springs" plus "Shock spring"."""
    labels, albedo, info = _scene()
    albedo = _albedo(labels, {0: GREY, 1: YELLOW, 2: YELLOW, 3: (70.0, 30.0, 60.0), 4: GOLD})    # the right one is orange
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    springs = next(g for g in groups if g.part)
    r2, g2, _ = grouping.split_group(groups, regions, gm, labels.copy(), albedo, springs.id, 2)
    names = sorted(g.name for g in g2 if g.part)
    assert names == ["Shock spring (left)", "Shock spring (right)"]
    springs.name = "Rear shocks"                                         # a custom name is kept by the piece that keeps the id
    r3, g3, _ = grouping.split_group(groups, regions, gm, labels.copy(), albedo, springs.id, 2)
    assert "Rear shocks" in {g.name for g in g3}


def test_instances_keep_the_parts_albedo_for_the_engine_until_their_regions_change():
    labels, albedo, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    springs = next(g for g in groups if g.part)
    r2, g2, gm2 = grouping.split_instances(groups, regions, gm, labels, springs.id)
    inst = [g for g in g2 if g.part]
    assert len(inst) == 2 and all(g.ref_lab == springs.albedo_lab for g in inst)
    assert all(g.ref_lab is None for g in g2 if not g.part)
    back = ColorGroup.from_dict(inst[0].to_dict())                        # it is stored with the job
    assert back.ref_lab == springs.albedo_lab
    frame = next(g for g in g2 if 1 in g.region_ids)
    r3, g3, _ = grouping.move_regions(g2, r2, gm2, labels, [4], frame.id)  # an unrelated move keeps it
    assert all(g.ref_lab == springs.albedo_lab for g in g3 if g.part)
    r4, g4, _ = grouping.merge_groups(g2, r2, gm2, labels, [g.id for g in inst])
    assert all(g.ref_lab is None for g in g4)                            # joined again: its own albedo
    left = next(g for g in g2 if g.name == "Shock spring (left)")
    r5, g5, _ = grouping.move_regions(g2, r2, gm2, labels, [4], left.id)   # its regions changed: dropped
    assert next(g for g in g5 if 2 in g.region_ids).ref_lab is None


def test_the_paint_family_leaves_out_a_minor_group_of_a_part():
    """A minor group whose parent is a part group is the part under other light: it goes with the
    part, not with the paint (painted with the paint, the robot's red feet got a navy rim)."""
    def grp(gid, lab, area, **kw):
        return ColorGroup(id=gid, name=f"g{gid}", albedo_lab=lab, albedo_hex="#c03020", area=area, area_frac=0.1,
                          region_ids=[gid], hue_family="red", **kw)

    groups = [grp(0, RED, 5000), grp(1, (45.0, 62.0, 45.0), 3000, part="foot", part_label="Foot", part_plural="Feet",
                                     part_instances=2),
              grp(2, (35.0, 40.0, 25.0), 100, minor=True, parent=1), grp(3, (40.0, 45.0, 35.0), 100, minor=True, parent=0)]
    assert refine.paint_family(groups) == [0, 3]
