"""
ShapeNet synset -> WordNet target mapping for "is YOLO right?" checks.

ShapeNet synset IDs are WordNet noun-offsets (e.g. 02958343 = car.n.01).
ImageNet classes are also WordNet synsets. ShapeNet's intended grouping is
typically broader than the WordNet synset itself (ShapeNet "car" includes
trucks/pickups, "airplane" includes warplanes/airships), so each synset is
mapped to a *target ancestor* — any prediction that's a hyponym of the
target counts as correct.

Used by `evaluate.py` and the optional correctness term in the training
reward.
"""

from __future__ import annotations

from functools import lru_cache

from nltk.corpus import wordnet as wn


# ShapeNet synset_id -> WordNet target ancestor.
# Predictions count as correct if they are this synset or a hyponym of it.
SYNSET_TO_TARGET: dict[str, str] = {
    "02691156": "aircraft.n.01",            # airplane: airliner / warplane / airship / space_shuttle
    "02958343": "motor_vehicle.n.01",       # car: car / truck / van / pickup / ambulance
    "03001627": "chair.n.01",
    "04379243": "table.n.01",
    "03211117": "display.n.06",             # monitor / screen / television
    "03046257": "timepiece.n.01",           # clock / watch
    "03642806": "personal_computer.n.01",   # laptop / notebook / desktop
    "03467517": "guitar.n.01",
    "03790512": "motor_scooter.n.01",       # motorcycle-ish; covers scooter / moped
    "03636649": "lamp.n.02",                # source of artificial light
    "04256520": "sofa.n.01",
    "03797390": "mug.n.04",                 # cup / coffee_mug
    "04090263": "gun.n.01",                 # rifle / assault_rifle / shotgun / revolver
    "04530566": "vessel.n.02",              # watercraft: ship / boat / canoe / kayak
    "02942699": "camera.n.01",              # digital_camera / reflex_camera / polaroid
}

SYNSET_TO_NAME: dict[str, str] = {
    "02691156": "airplane",
    "02958343": "car",
    "03001627": "chair",
    "04379243": "table",
    "03211117": "display",
    "03046257": "clock",
    "03642806": "laptop",
    "03467517": "guitar",
    "03790512": "motorcycle",
    "03636649": "lamp",
    "04256520": "sofa",
    "03797390": "mug",
    "04090263": "rifle",
    "04530566": "watercraft",
    "02942699": "camera",
}


@lru_cache(maxsize=2000)
def _synset_for_name(name: str):
    """Look up the first noun synset for an ImageNet class name (underscored)."""
    if not name:
        return None
    syns = wn.synsets(name, pos="n")
    return syns[0] if syns else None


@lru_cache(maxsize=128)
def _target_synset(shapenet_synset_id: str):
    target_name = SYNSET_TO_TARGET.get(shapenet_synset_id)
    return wn.synset(target_name) if target_name else None


def is_correct(shapenet_synset_id: str, predicted_idx: int, yolo_names: dict) -> bool:
    """True if the YOLO prediction is a hyponym of (or equal to) the
    target ancestor for this ShapeNet synset.

    `yolo_names` is the dict from `YOLO('weights').names` — `{idx: 'name'}`.
    """
    target = _target_synset(shapenet_synset_id)
    if target is None:
        return False
    pred = _synset_for_name(yolo_names.get(predicted_idx))
    if pred is None:
        return False
    return pred == target or target in pred.closure(lambda s: s.hypernyms())


def has_mapping(synset_id: str) -> bool:
    return synset_id in SYNSET_TO_TARGET
