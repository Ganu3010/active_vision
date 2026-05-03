"""
ShapeNetCore Dataset Loader
============================
Handles discovery and sampling of 3D mesh files from a local
ShapeNetCore directory (downloaded from HuggingFace).

Expected directory layout (standard ShapeNetCore v2):
    <root>/
        <synset_id>/           e.g. 02691156  (airplane)
            <model_id>/
                models/
                    model_normalized.obj
        taxonomy.json          (optional, used for category names)
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np


# Mapping of common ShapeNet synset IDs → human-readable names.
# Extended from the official ShapeNet taxonomy.
SYNSET_TO_CATEGORY: Dict[str, str] = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorbike",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "watercraft",
    "04554684": "washer",
}


Split = Literal["train", "val", "test", "all"]


def _bucket(synset_id: str, model_id: str) -> int:
    """Stable 0–99 bucket from (synset, model). Survives re-extract."""
    h = hashlib.md5(f"{synset_id}/{model_id}".encode()).hexdigest()
    return int(h[:8], 16) % 100


def _in_split(synset_id: str, model_id: str, split: Split) -> bool:
    if split == "all":
        return True
    b = _bucket(synset_id, model_id)
    if split == "train":
        return b < 70
    if split == "val":
        return 70 <= b < 85
    if split == "test":
        return b >= 85
    raise ValueError(f"unknown split: {split}")


class ShapeNetDataset:
    """Index and sample objects from a local ShapeNetCore directory.

    Parameters
    ----------
    root:
        Root of the ShapeNetCore dataset directory.
    categories:
        Optional list of synset IDs or category names to restrict sampling.
        If ``None``, all available categories are used.
    mesh_filename:
        Name of the OBJ file inside each model's ``models/`` subdirectory.
    """

    def __init__(
        self,
        root: Path | str,
        categories: Optional[List[str]] = None,
        mesh_filename: str = "model_normalized.obj",
        split: Split = "all",
    ):
        self.root = Path(root)
        self.mesh_filename = mesh_filename
        self.split: Split = split
        self._taxonomy = self._load_taxonomy()

        # Build the object index
        self._objects: List[Dict] = []
        self._build_index(categories)

        if not self._objects:
            raise RuntimeError(
                f"No ShapeNet objects found under {self.root}. "
                "Check that the dataset is downloaded and the path is correct."
            )

    # ------------------------------------------------------------------
    def _load_taxonomy(self) -> Dict[str, str]:
        """Load synset → name mapping from taxonomy.json if available."""
        taxonomy_path = self.root / "taxonomy.json"
        mapping = dict(SYNSET_TO_CATEGORY)  # start from built-in mapping
        if taxonomy_path.exists():
            try:
                with open(taxonomy_path) as f:
                    data = json.load(f)
                for entry in data:
                    sid = entry.get("synsetId", "")
                    name = entry.get("name", "").split(",")[0].strip()
                    if sid and name:
                        mapping[sid] = name
            except Exception:
                pass  # fall back to built-in mapping
        return mapping

    def _build_index(self, categories: Optional[List[str]]):
        """Walk the dataset directory and index all available meshes."""
        # Normalise category filter to a set of synset IDs
        allowed_synsets: Optional[set] = None
        if categories is not None:
            allowed_synsets = set()
            cat_to_synset = {v: k for k, v in self._taxonomy.items()}
            for c in categories:
                if c in self._taxonomy:
                    allowed_synsets.add(c)
                elif c in cat_to_synset:
                    allowed_synsets.add(cat_to_synset[c])
                else:
                    # Try partial match
                    for name, sid in cat_to_synset.items():
                        if c.lower() in name.lower():
                            allowed_synsets.add(sid)

        for synset_dir in sorted(self.root.iterdir()):
            if not synset_dir.is_dir():
                continue
            synset_id = synset_dir.name
            if allowed_synsets is not None and synset_id not in allowed_synsets:
                continue

            category = self._taxonomy.get(synset_id, synset_id)

            for model_dir in synset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                mesh_path = model_dir / "models" / self.mesh_filename
                if not mesh_path.exists():
                    # Try alternate layouts
                    alt = model_dir / self.mesh_filename
                    if alt.exists():
                        mesh_path = alt
                    else:
                        continue

                if not _in_split(synset_id, model_dir.name, self.split):
                    continue

                self._objects.append(
                    {
                        "synset_id": synset_id,
                        "model_id": model_dir.name,
                        "category": category,
                        "mesh_path": mesh_path,
                    }
                )

    # ------------------------------------------------------------------
    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict:
        """Return a random object metadata dict."""
        if rng is not None:
            idx = int(rng.integers(0, len(self._objects)))
        else:
            idx = random.randrange(len(self._objects))
        return self._objects[idx]

    def get_by_synset(self, synset_id: str) -> List[Dict]:
        return [o for o in self._objects if o["synset_id"] == synset_id]

    def get_by_category(self, category: str) -> List[Dict]:
        return [o for o in self._objects if o["category"] == category]

    @property
    def num_objects(self) -> int:
        return len(self._objects)

    @property
    def categories(self) -> List[str]:
        return sorted({o["category"] for o in self._objects})

    @property
    def synset_ids(self) -> List[str]:
        return sorted({o["synset_id"] for o in self._objects})

    def __len__(self) -> int:
        return len(self._objects)

    def __repr__(self) -> str:
        return (
            f"ShapeNetDataset(root={self.root}, "
            f"objects={self.num_objects}, "
            f"categories={len(self.categories)})"
        )
