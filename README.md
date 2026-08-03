# Ìtàn

Offline agricultural extension agent — retrieval-grounded, tool-calling, running fully
on-device via llama.cpp + Qwen2.5-1.5B-Instruct. Built for the Africa Deep Tech Challenge 2026
(Agriculture track). Full design rationale: `Itan_ADTC2026_Blueprint.pdf`.

## Repo layout

```
engine/       server.py (FastAPI orchestration), agent.py (router + tiers), retrieval.py
              (hybrid dense+BM25), language.py (lexicon), router/ (tier classifier),
              tools/agri_calc/ (deterministic fertiliser/seed/spray/margin math)
corpus/       ingestion pipeline (01-08_*.py) that BUILDS the RAG corpus. You should not
              need to run this -- see "Corpus data" below, pull the built artifacts instead.
eval/         gold-question dataset, accuracy/router/tool test harnesses
model/        where the downloaded .gguf weight file lands (gitignored, not committed)
metadata.json, download_model.sh, REPORT.md   ADTC submission-template required files
```

## Setup

**1. Clone and install Python deps** (Python 3.11+)
```bash
git clone <this-repo-url>
cd Itan-ADTC
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
(`corpus/requirements.txt` is separate and only needed if you're rebuilding the corpus from
raw sources -- you shouldn't need it, see below.)

**2. Corpus data (from Google Drive, not this repo)**

The built corpus artifacts are too large for git and are gitignored. Download them from
the shared Drive folder: **`<TODO: paste the Google Drive link here>`**

Place them at these exact paths (matching what `corpus/.gitignore` excludes):
```
corpus/chunks.parquet
corpus/vectors.npy
corpus/chunk_ids.json
corpus/bm25.pkl
corpus/structured.db
corpus/models/bge-small-en-v1.5-onnx/   (model.onnx, model_quantized.onnx, tokenizer files, config.json)
```
If a file is missing, `engine.retrieval.RetrievalEngine` degrades silently for that piece
(e.g. no BM25 index -> dense-only retrieval) rather than crashing, so a partial pull will
still run, just worse -- get all of them.

**3. Download the model weights**
```bash
bash download_model.sh
```
Idempotent, downloads `qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1.04GB) from Hugging Face to `model/`.

**4. Get llama.cpp** (provides `llama-server`, used for actual inference)

Not vendored in this repo. Either build from source or grab a prebuilt release for your OS
from https://github.com/ggml-org/llama.cpp/releases (look for `llama-b*-bin-<platform>.zip`,
CPU build is fine, no GPU needed). Extract it somewhere and note the path to `llama-server`
(`llama-server.exe` on Windows).

**5. Start the two servers** (two terminals)
```bash
# Terminal 1 -- the model
llama-server -m model/qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080 --host 127.0.0.1 -c 4096 -t 4

# Terminal 2 -- the orchestration layer (from repo root)
python -m uvicorn engine.server:app --host 127.0.0.1 --port 8000
```

**6. Test it**
```bash
curl -s -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended spacing for maize in the Northern Guinea Savanna?"}'
```
Or run the eval harness against a batch of gold questions:
```bash
python eval/run_eval.py --limit 20
```
This will refuse to run (loudly, on purpose) if `llama-server` isn't reachable -- it will not
silently fake results.

## Known issues (see REPORT.md / team for latest)

- The tier router currently misroutes a meaningful fraction of real-world-phrased questions
  to refusal (Tier D) -- it's well-calibrated on its own 200-question training set but doesn't
  generalize as well beyond it yet.
- Retrieval sometimes returns topically-adjacent-but-wrong documents for pest/disease
  questions (measured ~55% hit rate on the pest-diagnosis validation slice) -- corpus
  chunking has a long tail of oversized chunks that likely hurts match precision.
- No UI yet -- `/query` via HTTP is the only interface right now.
- Query latency is currently slow on constrained dev machines; not yet benchmarked on
  ADTC-representative hardware.
