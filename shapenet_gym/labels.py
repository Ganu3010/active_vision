"""
ShapeNet synset -> ImageNet class index mapping.

YOLOv8-cls outputs probabilities over the 1000 ImageNet classes; ShapeNet
ground truth uses synset IDs. This module bridges the two for evaluation:
given a ShapeNet object and a YOLO top-1 prediction, decide whether the
prediction is "correct."

Used by `evaluate.py` only — never imported by `train_ppo.py` or the env.
The training reward is entropy-only and does not consult these labels.

A ShapeNet synset maps to a *set* of ImageNet indices because most ShapeNet
categories correspond to multiple ImageNet sub-types (e.g., ShapeNet "car"
includes any of {sports_car, convertible, limousine, ...}).

ImageNet indices verified against torchvision's ImageNet1k labels.
"""

from __future__ import annotations


# ShapeNet synset_id -> set of ImageNet-1k class indices that count as "correct"
SYNSET_TO_IMAGENET_INDICES: dict[str, set[int]] = {
    # 02691156 airplane
    "02691156": {404, 895, 812},          # airliner, warplane, space_shuttle

    # 02958343 car
    "02958343": {817, 511, 627, 751, 468, 656, 609, 717, 661},
    # sports_car, convertible, limousine, racer, cab, minivan, jeep, pickup, Model_T

    # 03001627 chair
    "03001627": {857, 559, 423, 765},     # throne, folding_chair, barber_chair, rocking_chair

    # 04379243 table
    "04379243": {532, 526},               # dining_table, desk

    # 03211117 display / monitor
    "03211117": {664, 782, 851},          # monitor, screen, television

    # 03046257 clock
    "03046257": {409, 530, 892},          # analog_clock, digital_clock, wall_clock

    # 03642806 laptop
    "03642806": {620, 681},               # laptop, notebook

    # 03467517 guitar
    "03467517": {402, 546},               # acoustic_guitar, electric_guitar

    # 03790512 motorcycle
    "03790512": {670, 665},               # motor_scooter, moped

    # 03636649 lamp
    "03636649": {846, 818},               # table_lamp, spotlight

    # 04256520 sofa
    "04256520": {831},                    # studio_couch

    # 03797390 mug
    "03797390": {504, 968},               # coffee_mug, cup
}


# Human-readable names — for log/eval output
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
}


def is_correct(synset_id: str, predicted_imagenet_idx: int) -> bool:
    """Return True if the YOLO prediction is one of the accepted ImageNet
    classes for the given ShapeNet synset.

    Synsets without a defined mapping return False — by design, callers
    should filter those out before calling is_correct(), or accept that
    those synsets will count as 0% accuracy.
    """
    return predicted_imagenet_idx in SYNSET_TO_IMAGENET_INDICES.get(synset_id, set())


def has_mapping(synset_id: str) -> bool:
    """Whether this synset has any ImageNet correspondents defined."""
    return synset_id in SYNSET_TO_IMAGENET_INDICES
