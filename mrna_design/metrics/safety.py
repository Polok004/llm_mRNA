"""
Off-target safety metrics.

Detects potential off-target interactions of an mRNA candidate:

1. miRNA seed-match scanner (7mer-m8 and 8mer) — no external tool needed.
2. RNAhybrid subprocess wrapper — thermodynamic miRNA binding prediction.
3. BLAST+ subprocess wrapper (blastn-short) — homology to human transcriptome.

Database paths are configured via environment variables or mrna_design config:
  MIRBASE_DB   — path to miRBase hsa mature miRNA FASTA (or BLAST DB prefix)
  GENCODE_DB   — path to GENCODE CDS BLAST DB prefix

Both databases are set up by scripts/setup_databases.sh.

Seeed-match logic (Grimson et al. 2007)
----------------------------------------
  8mer  : nt 2-8 of miRNA match + A at position 1 of target
  7mer-m8: nt 2-8 of miRNA match (no A requirement at pos 1)
  7mer-A1: nt 2-7 of miRNA match + A at position 1
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from mrna_design.logging_utils import get_logger

log = get_logger("metrics.safety")

# ── Config ────────────────────────────────────────────────────────────────────

_MIRBASE_FASTA = Path(os.environ.get(
    "MIRBASE_DB",
    str(Path(__file__).parent.parent.parent / "data" / "mirbase" / "hsa_mature.fa"),
))
_GENCODE_DB = Path(os.environ.get(
    "GENCODE_DB",
    str(Path(__file__).parent.parent.parent / "data" / "gencode" / "cds_blast_db"),
))

_BLAST_EVALUE = 0.01
_BLAST_WORD_SIZE = 7    # appropriate for miRNA-length short sequences
_RNAHYBRID_TIMEOUT = 30  # seconds per call


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class MirnaSeedHit:
    mirna_id: str
    seed_type: str          # "8mer", "7mer-m8", "7mer-A1"
    target_start: int       # 0-based nt in the provided `target_region` string
    target_end: int
    seed_sequence: str      # the seed (nt 2-8 of miRNA)

@dataclass
class RnahybridHit:
    mirna_id: str
    delta_g: float
    target_start: int
    target_end: int

@dataclass
class BlastHit:
    query_id: str
    subject_id: str
    pct_identity: float
    alignment_len: int
    evalue: float
    bitscore: float

@dataclass
class SafetyResult:
    mirna_seed_hits: list[MirnaSeedHit] = field(default_factory=list)
    rnahybrid_hits: list[RnahybridHit] = field(default_factory=list)
    blast_hits: list[BlastHit] = field(default_factory=list)

    @property
    def total_mirna_hits(self) -> int:
        return len(self.mirna_seed_hits)

    @property
    def total_blast_hits(self) -> int:
        return len(self.blast_hits)


# ── miRNA seed scanner ────────────────────────────────────────────────────────

def _load_mirna_seeds(fasta_path: Path) -> dict[str, str]:
    """
    Load human mature miRNA sequences from a FASTA file.
    Returns {mirna_id: sequence} (RNA, uppercase).
    Falls back to an empty dict if file not found.
    """
    if not fasta_path.exists():
        log.warn(
            "mirbase_fasta_not_found",
            path=str(fasta_path),
            hint="Run: make db",
        )
        return {}
    mirnas: dict[str, str] = {}
    current_id = ""
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:].split()[0]
            else:
                mirnas[current_id] = mirnas.get(current_id, "") + line.upper().replace("T", "U")
    return mirnas


# Lazy-load the miRNA seed library once per process
_MIRNA_CACHE: dict[str, str] | None = None


def _get_mirnas() -> dict[str, str]:
    global _MIRNA_CACHE
    if _MIRNA_CACHE is None:
        _MIRNA_CACHE = _load_mirna_seeds(_MIRBASE_FASTA)
    return _MIRNA_CACHE


def scan_mirna_seeds(
    target: str,
    region_offset: int = 0,
    min_seed_type: str = "7mer-m8",
) -> list[MirnaSeedHit]:
    """
    Scan `target` for miRNA seed matches.

    Parameters
    ----------
    target : str
        RNA target sequence (CDS + 3'UTR or just 3'UTR).
    region_offset : int
        Offset of `target` within the full candidate sequence (for coordinates).
    min_seed_type : "8mer" | "7mer-m8" | "7mer-A1"
        Minimum seed stringency. "7mer-m8" is the standard minimum.
    """
    target = target.upper().replace("T", "U")
    mirnas = _get_mirnas()
    hits: list[MirnaSeedHit] = []

    seed_types = {
        "8mer": True,
        "7mer-m8": True,
        "7mer-A1": min_seed_type != "8mer",
    }
    if min_seed_type == "8mer":
        seed_types["7mer-m8"] = False
        seed_types["7mer-A1"] = False

    for mirna_id, mirna_seq in mirnas.items():
        if len(mirna_seq) < 8:
            continue
        # Seed = nt 2-8 of miRNA (1-indexed → indices 1:8 in Python)
        seed_7 = mirna_seq[1:7]   # nt 2-7  (6-mer)
        seed_8 = mirna_seq[1:8]   # nt 2-8  (7-mer)

        # miRNA binds in antiparallel — we look for reverse complement of seed in target
        from mrna_design.metrics.immunogenicity import _reverse_complement

        rc_seed_7 = _reverse_complement(seed_7)
        rc_seed_8 = _reverse_complement(seed_8)

        # 8mer: rc of nt2-8 matches + A at target position before match
        if seed_types["8mer"]:
            for m in re.finditer(re.escape(rc_seed_8), target):
                pos = m.start()
                if pos > 0 and target[pos - 1] == "A":
                    hits.append(MirnaSeedHit(
                        mirna_id=mirna_id,
                        seed_type="8mer",
                        target_start=region_offset + pos - 1,
                        target_end=region_offset + pos + len(rc_seed_8),
                        seed_sequence=rc_seed_8,
                    ))

        # 7mer-m8: rc of nt2-8 matches (no A requirement)
        if seed_types["7mer-m8"]:
            for m in re.finditer(re.escape(rc_seed_8), target):
                pos = m.start()
                hits.append(MirnaSeedHit(
                    mirna_id=mirna_id,
                    seed_type="7mer-m8",
                    target_start=region_offset + pos,
                    target_end=region_offset + pos + len(rc_seed_8),
                    seed_sequence=rc_seed_8,
                ))

        # 7mer-A1: rc of nt2-7 matches + A at target position before match
        if seed_types["7mer-A1"]:
            for m in re.finditer(re.escape(rc_seed_7), target):
                pos = m.start()
                if pos > 0 and target[pos - 1] == "A":
                    hits.append(MirnaSeedHit(
                        mirna_id=mirna_id,
                        seed_type="7mer-A1",
                        target_start=region_offset + pos - 1,
                        target_end=region_offset + pos + len(rc_seed_7),
                        seed_sequence=rc_seed_7,
                    ))

    # Deduplicate: same position and miRNA
    seen: set[tuple] = set()
    unique: list[MirnaSeedHit] = []
    for h in hits:
        key = (h.mirna_id, h.target_start, h.seed_type)
        if key not in seen:
            seen.add(key)
            unique.append(h)

    log.event("mirna_scan_done", n_hits=len(unique), target_len=len(target))
    return unique


# ── RNAhybrid wrapper ─────────────────────────────────────────────────────────

def run_rnahybrid(
    target: str,
    mirna_fasta: Path | None = None,
    max_hits: int = 20,
    min_delta_g: float = -20.0,
) -> list[RnahybridHit]:
    """
    Run RNAhybrid for thermodynamic miRNA binding prediction.

    Returns an empty list if RNAhybrid is not installed.
    """
    mirna_fasta = mirna_fasta or _MIRBASE_FASTA
    if not mirna_fasta.exists():
        log.warn("rnahybrid_mirna_fasta_missing", path=str(mirna_fasta))
        return []

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as tf:
            tf.write(f">query\n{target}\n")
            tf_path = tf.name

        proc = subprocess.run(
            [
                "RNAhybrid",
                "-t", tf_path,
                "-q", str(mirna_fasta),
                "-e", str(min_delta_g),
                "-m", str(max_hits),
                "-s", "3utr_human",
            ],
            capture_output=True,
            text=True,
            timeout=_RNAHYBRID_TIMEOUT,
        )
        hits: list[RnahybridHit] = []
        if proc.returncode == 0:
            # Parse: lines starting with "target:" contain target name,
            # lines with "mfe:" contain the ΔG, lines with "position:" contain start
            current_mirna = ""
            current_dg = 0.0
            current_pos = 0
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("miRNA"):
                    current_mirna = line.split()[-1] if line.split() else ""
                elif line.startswith("mfe:"):
                    try:
                        current_dg = float(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("position"):
                    try:
                        current_pos = int(line.split()[-1])
                    except (IndexError, ValueError):
                        pass
                    if current_mirna:
                        hits.append(RnahybridHit(
                            mirna_id=current_mirna,
                            delta_g=current_dg,
                            target_start=current_pos,
                            target_end=current_pos + 21,  # approx miRNA length
                        ))
        log.event("rnahybrid_done", n_hits=len(hits))
        return hits

    except FileNotFoundError:
        log.warn("rnahybrid_not_installed", hint="Install RNAhybrid or skip.")
        return []
    except subprocess.TimeoutExpired:
        log.warn("rnahybrid_timeout")
        return []
    finally:
        import os as _os
        try:
            _os.unlink(tf_path)
        except Exception:
            pass


# ── BLAST+ wrapper ────────────────────────────────────────────────────────────

def run_blast_short(
    query_seq: str,
    db_path: Path | None = None,
    evalue: float = _BLAST_EVALUE,
    max_hits: int = 10,
) -> list[BlastHit]:
    """
    Run blastn-short against a local BLAST+ database.

    Returns an empty list if BLAST+ is not installed or db not found.
    """
    db_path = db_path or _GENCODE_DB
    if not Path(str(db_path) + ".nhr").exists():
        log.warn("blast_db_not_found", path=str(db_path), hint="Run: make db")
        return []

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as tf:
            tf.write(f">query\n{query_seq}\n")
            tf_path = tf.name

        proc = subprocess.run(
            [
                "blastn",
                "-task", "blastn-short",
                "-db", str(db_path),
                "-query", tf_path,
                "-evalue", str(evalue),
                "-max_target_seqs", str(max_hits),
                "-outfmt", "6 qseqid sseqid pident length evalue bitscore",
                "-word_size", str(_BLAST_WORD_SIZE),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        hits: list[BlastHit] = []
        if proc.returncode == 0:
            for line in proc.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 6:
                    try:
                        hits.append(BlastHit(
                            query_id=parts[0],
                            subject_id=parts[1],
                            pct_identity=float(parts[2]),
                            alignment_len=int(parts[3]),
                            evalue=float(parts[4]),
                            bitscore=float(parts[5]),
                        ))
                    except ValueError:
                        continue
        log.event("blast_done", n_hits=len(hits))
        return hits

    except FileNotFoundError:
        log.warn("blast_not_installed", hint="Install NCBI BLAST+")
        return []
    except subprocess.TimeoutExpired:
        log.warn("blast_timeout")
        return []
    finally:
        import os as _os
        try:
            _os.unlink(tf_path)
        except Exception:
            pass


# ── Aggregate ─────────────────────────────────────────────────────────────────

def compute_safety(
    sequence: str,
    cds_start: int,
    cds_end: int,
    utr3: str = "",
    run_rnahybrid_flag: bool = False,
    run_blast_flag: bool = True,
) -> SafetyResult:
    """
    Run all safety checks and return a SafetyResult.

    miRNA seed scanning is always run.
    RNAhybrid is optional (slow; run_rnahybrid_flag=False by default).
    BLAST is run if the db exists.
    """
    # Scan CDS + 3'UTR for miRNA seeds
    cds_and_3utr = sequence[cds_start:] if not utr3 else sequence[cds_start:cds_end] + utr3
    seed_hits = scan_mirna_seeds(cds_and_3utr, region_offset=cds_start)

    rh_hits: list[RnahybridHit] = []
    if run_rnahybrid_flag:
        rh_hits = run_rnahybrid(cds_and_3utr)

    blast_hits: list[BlastHit] = []
    if run_blast_flag:
        blast_hits = run_blast_short(sequence)

    return SafetyResult(
        mirna_seed_hits=seed_hits,
        rnahybrid_hits=rh_hits,
        blast_hits=blast_hits,
    )
