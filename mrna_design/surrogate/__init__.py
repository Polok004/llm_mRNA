"""mrna_design/surrogate package — lightweight surrogate model."""

from mrna_design.surrogate.features import DIM, FEATURE_NAMES, extract, batch_extract
from mrna_design.surrogate.model import SurrogateModel, SurrogatePrediction
from mrna_design.surrogate.scorer import get_global_model, surrogate_score
from mrna_design.surrogate.openvaccine import OVRecord, load_openvaccine

__all__ = [
    "DIM", "FEATURE_NAMES", "extract", "batch_extract",
    "SurrogateModel", "SurrogatePrediction",
    "get_global_model", "surrogate_score",
    "OVRecord", "load_openvaccine",
]
