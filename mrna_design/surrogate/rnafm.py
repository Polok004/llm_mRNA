"""
RNA-FM surrogate feature extractor.

RNA-FM (Chen et al., 2022) is a 640M-parameter RNA foundation model.
We use it purely as a feature extractor: the mean-pooled hidden state of the
last transformer layer serves as a rich 640-dim sequence embedding.

This module is **optional**. Import guards ensure it fails gracefully when
`transformers` / `torch` are not installed or no GPU/CPU tensors are available.

Usage::

    from mrna_design.surrogate.rnafm import RNAFM_AVAILABLE, embed_sequence, embed_batch
    if RNAFM_AVAILABLE:
        emb = embed_sequence("AUGCGA...")  # np.ndarray shape (640,)
    else:
        emb = np.zeros(RNAFM_DIM)         # transparent fallback
"""

from __future__ import annotations

import numpy as np

# ── Optional-import guard ─────────────────────────────────────────────────────

RNAFM_AVAILABLE: bool = False
RNAFM_DIM: int = 640
_MODEL = None
_TOKENIZER = None
_DEVICE = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModel  # type: ignore

    RNAFM_AVAILABLE = True
except ImportError:
    pass


_MODEL_NAME = "multimolecule/rnafm"  # HuggingFace hub ID


def _ensure_loaded() -> None:
    """Lazy-load the RNA-FM model and tokeniser on first use."""
    global _MODEL, _TOKENIZER, _DEVICE

    if not RNAFM_AVAILABLE:
        raise RuntimeError(
            "RNA-FM requires 'transformers' and 'torch'. "
            "Install with: pip install 'mrna-design[rnafm]'"
        )

    if _MODEL is None:
        import torch
        from transformers import AutoTokenizer, AutoModel

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)
        _MODEL = AutoModel.from_pretrained(_MODEL_NAME, trust_remote_code=True)
        _MODEL.eval()
        _MODEL.to(_DEVICE)


def embed_sequence(sequence: str) -> np.ndarray:
    """
    Compute the RNA-FM mean-pool embedding for a single RNA sequence.

    Parameters
    ----------
    sequence : str
        RNA sequence (any alphabet — U or T accepted; case insensitive).

    Returns
    -------
    np.ndarray of shape (640,), dtype float32.
    """
    return embed_batch([sequence])[0]


def embed_batch(sequences: list[str], batch_size: int = 8) -> np.ndarray:
    """
    Compute RNA-FM embeddings for a batch of sequences.

    Parameters
    ----------
    sequences : list[str]
        List of RNA sequences.
    batch_size : int
        Sequences per GPU forward pass (tune based on available VRAM).

    Returns
    -------
    np.ndarray of shape (N, 640), dtype float32.
    """
    _ensure_loaded()

    import torch

    # Normalise to RNA alphabet
    seqs = [s.upper().replace("T", "U") for s in sequences]

    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        inputs = _TOKENIZER(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=1024)
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _MODEL(**inputs)

        # outputs.last_hidden_state: (B, L, 640)
        hidden = outputs.last_hidden_state  # (B, L, 640)

        # Mean-pool over sequence positions (excluding padding tokens)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        sum_hidden = (hidden * attention_mask).sum(dim=1)               # (B, 640)
        lengths = attention_mask.sum(dim=1)                              # (B, 1)
        mean_hidden = (sum_hidden / lengths).cpu().numpy().astype(np.float32)  # (B, 640)

        all_embeddings.append(mean_hidden)

    return np.concatenate(all_embeddings, axis=0)
