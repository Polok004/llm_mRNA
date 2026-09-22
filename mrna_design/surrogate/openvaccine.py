"""
OpenVaccine dataset loader and target engineering.

The OpenVaccine Kaggle challenge (https://www.kaggle.com/c/stanford-covid-vaccine)
provides per-nucleotide experimental measurements for 3029 RNA constructs:
  - reactivity        (SHAPE-based, proxy for structure flexibility)
  - deg_Mg_pH10       (degradation in Mg²⁺ at pH 10)
  - deg_Mg_50C        (degradation in Mg²⁺ at 50°C)
  - deg_pH10          (degradation at pH 10, no Mg)
  - deg_50C           (degradation at 50°C, no Mg)

We aggregate these to two per-sequence targets useful for our surrogate:
  1. mean_reactivity     → proxy for solvent accessibility (higher = more flexible)
  2. mean_degradation    → mean of the four degradation columns

The dataset JSON is expected at:
  data/openvaccine/train.json   (downloaded by scripts/setup_databases.sh)

If the file does not exist we provide a synthetic dummy dataset for CI tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


_DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "openvaccine" / "train.json"

# Per-nt columns to average for the degradation target
_DEG_COLS = ["deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]


@dataclass
class OVRecord:
    """One OpenVaccine training record."""
    sequence_id: str
    sequence: str          # RNA sequence (A/C/G/U), 107 nt
    cds_start: int         # always 0 for the 107-nt constructs (no 5'UTR)
    cds_end: int
    mean_reactivity: float
    mean_degradation: float


def load_openvaccine(
    path: Path | str | None = None,
    max_records: int | None = None,
) -> list[OVRecord]:
    """
    Load the OpenVaccine training JSON and return parsed OVRecord objects.

    Parameters
    ----------
    path : Path | str | None
        Path to train.json. Falls back to the default location.
    max_records : int | None
        Limit the number of records loaded (useful for tests).

    Returns
    -------
    list[OVRecord]
    """
    path = Path(path) if path else _DEFAULT_DATA_PATH

    if not path.exists():
        return _synthetic_dataset(n=200)

    with open(path) as f:
        raw = json.load(f)

    records = raw if isinstance(raw, list) else raw.get("train_data", raw.get("data", []))
    if max_records:
        records = records[:max_records]

    result = []
    for r in records:
        seq = r.get("sequence", "").upper().replace("T", "U")
        if not seq:
            continue

        # Reactivity column is a list of per-nt values
        reactivity = r.get("reactivity", [])
        mean_react = float(np.nanmean(reactivity)) if reactivity else 0.0

        # Degradation columns
        deg_vals = []
        for col in _DEG_COLS:
            col_data = r.get(col, [])
            if col_data:
                deg_vals.extend([v for v in col_data if v is not None])
        mean_deg = float(np.nanmean(deg_vals)) if deg_vals else 0.0

        result.append(OVRecord(
            sequence_id=r.get("id", r.get("sequence_id", f"ov_{len(result)}")),
            sequence=seq,
            cds_start=0,
            cds_end=len(seq),
            mean_reactivity=mean_react,
            mean_degradation=mean_deg,
        ))

    return result


def _synthetic_dataset(n: int = 200, rng_seed: int = 0) -> list[OVRecord]:
    """
    Generate a deterministic synthetic dataset for CI / offline tests.
    Sequences are random 107-nt RNA; targets are noisy functions of GC content
    to give the surrogate something real to learn.
    """
    import random
    rng = random.Random(rng_seed)
    bases = ["A", "C", "G", "U"]
    records = []
    for i in range(n):
        seq = "".join(rng.choices(bases, weights=[0.3, 0.2, 0.2, 0.3], k=107))
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        # Reactivity: correlated negatively with GC (high GC → more structured → less reactive)
        react = max(0.0, 0.7 - gc + rng.gauss(0, 0.05))
        # Degradation: correlated positively with reactivity and length
        deg = max(0.0, 0.4 + 0.3 * react + rng.gauss(0, 0.03))
        records.append(OVRecord(
            sequence_id=f"synthetic_{i:04d}",
            sequence=seq,
            cds_start=0,
            cds_end=len(seq),
            mean_reactivity=react,
            mean_degradation=deg,
        ))
    return records


def records_to_arrays(
    records: list[OVRecord],
    feature_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of OVRecords into (X, y_reactivity, y_degradation) arrays.

    Parameters
    ----------
    records : list[OVRecord]
    feature_fn : callable
        e.g. mrna_design.surrogate.features.extract

    Returns
    -------
    X               : np.ndarray (N, DIM)
    y_reactivity    : np.ndarray (N,)
    y_degradation   : np.ndarray (N,)
    """
    Xs = []
    y_react = []
    y_deg = []
    for r in records:
        try:
            feat = feature_fn(r.sequence, r.cds_start, r.cds_end, utr5="")
            Xs.append(feat)
            y_react.append(r.mean_reactivity)
            y_deg.append(r.mean_degradation)
        except Exception:
            continue
    return (
        np.array(Xs, dtype=np.float32),
        np.array(y_react, dtype=np.float32),
        np.array(y_deg, dtype=np.float32),
    )
