#!/bin/bash
# Full pipeline: harvest metadata -> score -> gap analysis -> manifest ->
# NAFDAC extract -> auto-download -> extract -> clean -> chunk -> structure
# extract -> embed -> BM25 -> validate.
#
# Stops on the first failing stage. Every stage prints a timestamped start
# and end log line so progress is visible at all times.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-python}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_stage() {
    local name="$1"
    local script="$2"
    log "START  $name"
    if ! "$PY" "$script"; then
        log "FAILED $name -- stopping pipeline"
        exit 1
    fi
    log "DONE   $name"
}

log "=== ITAN CORPUS PIPELINE START ==="

run_stage "harvest/01_discover"        "harvest/01_discover.py"
run_stage "harvest/02_score"           "harvest/02_score.py"
run_stage "harvest/03_gap_analysis"    "harvest/03_gap_analysis.py"
run_stage "harvest/04_manifest"        "harvest/04_manifest.py"
run_stage "harvest/05_nafdac_extract"  "harvest/05_nafdac_extract.py"

run_stage "01_fetch"                   "01_fetch.py"
run_stage "02_extract"                 "02_extract.py"
run_stage "03_clean"                   "03_clean.py"
run_stage "04_chunk"                   "04_chunk.py"
run_stage "05_structure_extract"       "05_structure_extract.py"
run_stage "06_embed"                   "06_embed.py"
run_stage "07_bm25"                    "07_bm25.py"
run_stage "08_validate"                "08_validate.py"

log "=== ITAN CORPUS PIPELINE COMPLETE ==="
echo ""
echo "Next steps:"
echo "  - review harvest/gap_report.txt for crop/zone combinations that still need manual sourcing"
echo "  - review corpus/structured.db rows flagged needs_review=1 before using them live"
echo "  - review corpus/validation_report.json for retrieval hit-rate gaps"
echo "  - harvest/05_nafdac_extract.py already loaded harvest/nafdac_agrochemicals.csv into the agrochemical table (it bypasses 04_chunk.py)"
