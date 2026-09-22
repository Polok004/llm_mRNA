#!/usr/bin/env bash
set -euo pipefail

# Scripts for downloading and formatting databases required by the mRNA pipeline.
# Run this from the repository root: `make db`

DATA_DIR="data"
MIRBASE_DIR="${DATA_DIR}/mirbase"
GENCODE_DIR="${DATA_DIR}/gencode"

mkdir -p "$MIRBASE_DIR"
mkdir -p "$GENCODE_DIR"

# ── 1. miRBase (Human Mature miRNAs) ──────────────────────────────────────────

MIRBASE_FA="${MIRBASE_DIR}/hsa_mature.fa"

if [[ ! -f "$MIRBASE_FA" ]]; then
    echo "Downloading miRBase human mature miRNAs..."
    # Download the global mature.fa
    curl -sL "ftp://mirbase.org/pub/mirbase/CURRENT/mature.fa.gz" -o "${MIRBASE_DIR}/mature.fa.gz"
    
    # Extract only human (hsa) miRNAs
    echo "Extracting hsa (Homo sapiens) sequences..."
    gunzip -c "${MIRBASE_DIR}/mature.fa.gz" | \
        awk '
        /^>/ {
            if ($1 ~ />hsa-/) {
                keep = 1
                print $0
            } else {
                keep = 0
            }
            next
        }
        {
            if (keep) print $0
        }
        ' > "$MIRBASE_FA"
        
    rm "${MIRBASE_DIR}/mature.fa.gz"
    echo "Saved to $MIRBASE_FA"
else
    echo "miRBase FASTA already exists at $MIRBASE_FA"
fi


# ── 2. GENCODE CDS (Human) ────────────────────────────────────────────────────

GENCODE_FA="${GENCODE_DIR}/gencode_cds.fa"
BLAST_DB="${GENCODE_DIR}/cds_blast_db"

if [[ ! -f "${BLAST_DB}.nhr" ]]; then
    echo "Downloading GENCODE Human CDS transcripts (Release 45)..."
    # Note: Use a stable release version url. Adjust if needed.
    GENCODE_URL="ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.pc_translations.fa.gz"
    
    curl -sL "$GENCODE_URL" -o "${GENCODE_DIR}/gencode.fa.gz"
    
    echo "Decompressing GENCODE FASTA..."
    gunzip -c "${GENCODE_DIR}/gencode.fa.gz" > "$GENCODE_FA"
    rm "${GENCODE_DIR}/gencode.fa.gz"
    
    echo "Building BLAST+ database..."
    if command -v makeblastdb &> /dev/null; then
        makeblastdb -in "$GENCODE_FA" -dbtype nucl -out "$BLAST_DB" -title "GENCODE_Human_CDS"
        echo "BLAST DB built at $BLAST_DB"
    else
        echo "WARNING: makeblastdb not found on PATH. BLAST database not built."
        echo "Please install NCBI BLAST+ (e.g., 'brew install blast' or 'apt install ncbi-blast+')"
        echo "Then run: makeblastdb -in $GENCODE_FA -dbtype nucl -out $BLAST_DB"
    fi
else
    echo "GENCODE BLAST DB already exists at ${BLAST_DB}.nhr"
fi

echo "Database setup complete."
