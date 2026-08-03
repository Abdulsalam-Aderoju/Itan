#!/usr/bin/env python
"""
Stage 6 (build): dense embeddings for every chunk.

Model: BAAI/bge-small-en-v1.5, exported to ONNX and dynamically quantized
to 8-bit via optimum + onnxruntime (CPU-friendly, no GPU required).

Output: corpus/vectors.npy (float16, shape = [n_chunks, hidden_dim]) and
corpus/chunk_ids.json (index position -> chunk_id).

Runnable standalone: python corpus/06_embed.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS_DIR, print_elapsed_checkpoint, progress_line  # noqa: E402

# torch must be imported before numpy/pandas, not after -- on Windows,
# whichever of {torch, pandas} loads its native runtime DLLs into the
# process first wins a naming collision between their bundled copies.
# pandas-then-torch reliably raises OSError WinError 1114 loading torch's
# c10.dll; torch-then-pandas doesn't. transformers/optimum (which pull in
# torch transitively) are still imported lazily inside
# get_quantized_model_and_tokenizer() -- only paid for when actually
# embedding -- but by then torch is already resident so that transitive
# `from torch import Tensor` doesn't retrigger the collision.
import torch  # noqa: F401,E402

import numpy as np
import pandas as pd

CHUNKS_PARQUET = CORPUS_DIR / "chunks.parquet"
VECTORS_NPY = CORPUS_DIR / "vectors.npy"
CHUNK_IDS_JSON = CORPUS_DIR / "chunk_ids.json"
# A full-corpus embed run can take hours; these hold in-progress state so a
# crash or interrupt loses at most one batch instead of the whole run. See
# run_embed_corpus()/save_partial()/load_partial_progress() below. Deleted
# once a run finishes successfully -- their presence always means "resumable
# in-progress state", never "the finished output" (that's VECTORS_NPY).
PARTIAL_VECTORS_NPY = CORPUS_DIR / "vectors.partial.npy"
PARTIAL_CHUNK_IDS_JSON = CORPUS_DIR / "chunk_ids.partial.json"
ONNX_DIR = CORPUS_DIR / "models" / "bge-small-en-v1.5-onnx"

MODEL_ID = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64
MAX_LENGTH = 512


def get_quantized_model_and_tokenizer():
    """Export bge-small-en-v1.5 to ONNX (once, cached under corpus/models/)
    and dynamically quantize it to 8-bit. Returns (ort_model, tokenizer).

    Every step here prints before it starts, not just after it finishes --
    this whole function runs before the per-batch progress loop in main()
    even begins, and on a first run (model download + ONNX export +
    quantization) it can silently sit for minutes on a slow link or a slow
    CPU. Without a print per step, a stall here and a crash here look
    identical from the terminal -- pinning down the failing step (e.g. from
    a traceback) is worthless if you can't first tell it stalled."""
    print(f"[embed] importing transformers/optimum (can be slow on first import) ...")
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    quantized_path = ONNX_DIR / "model_quantized.onnx"

    print(f"[embed] loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if not quantized_path.exists():
        ONNX_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[embed] exporting {MODEL_ID} to ONNX (first run only, downloads the base model) ...")
        base_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
        base_model.save_pretrained(ONNX_DIR)
        tokenizer.save_pretrained(ONNX_DIR)
        print(f"[embed] ONNX export done -> {ONNX_DIR}")

        print("[embed] applying dynamic 8-bit quantization ...")
        quantizer = ORTQuantizer.from_pretrained(ONNX_DIR)
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=ONNX_DIR, quantization_config=qconfig)
        print("[embed] quantization done")
    else:
        print(f"[embed] reusing cached quantized model at {quantized_path}")

    print("[embed] loading quantized ONNX model into onnxruntime ...")
    model = ORTModelForFeatureExtraction.from_pretrained(ONNX_DIR, file_name="model_quantized.onnx")
    print("[embed] model + tokenizer ready")
    return model, tokenizer


def mean_pool_or_cls(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # bge models are typically used with CLS-token pooling.
    cls = last_hidden_state[:, 0, :]
    norm = np.linalg.norm(cls, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    return cls / norm


def _embed_batch(batch: list[str], model, tokenizer) -> np.ndarray:
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="np")
    outputs = model(**inputs)
    last_hidden = np.asarray(outputs.last_hidden_state) if hasattr(outputs, "last_hidden_state") else np.asarray(outputs[0])
    return mean_pool_or_cls(last_hidden, inputs["attention_mask"])


def embed_texts(
    texts: list[str], model=None, tokenizer=None, batch_size: int = BATCH_SIZE, stage_label: str | None = None,
) -> np.ndarray:
    """Embed a list of texts with the quantized bge model. Lazily loads the
    model if not provided (used standalone by 08_validate.py for queries).

    stage_label: when set (only by main()'s full-corpus run), prints one
    progress line per BATCH rather than per chunk -- embedding one chunk at
    a time would serialize the ONNX call and defeat the entire point of
    BATCH_SIZE, so the batch is the smallest unit progress is reported at
    here. Left None for the single-query embed calls made by 08_validate.py
    so those don't spam batch-of-1 progress lines.

    No checkpointing here -- this is also called for small single-query
    embeds where it would be pure overhead. main()'s full-corpus run uses
    its own checkpointed loop below instead, since a run over the whole
    corpus can take hours and losing all of it to a crash near the end is
    a real risk worth guarding against (see run_embed_corpus())."""
    if model is None or tokenizer is None:
        model, tokenizer = get_quantized_model_and_tokenizer()

    all_vecs = []
    n_batches = max(1, -(-len(texts) // batch_size))  # ceil division
    start_time = time.time()
    for bi, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i:i + batch_size]
        label = f"chunks {i + 1}-{min(i + batch_size, len(texts))}"
        try:
            vecs = _embed_batch(batch, model, tokenizer)
            all_vecs.append(vecs)
            if stage_label:
                progress_line(stage_label, bi, n_batches, label, ok=True)
        except Exception as exc:
            if stage_label:
                progress_line(stage_label, bi, n_batches, label, ok=False, reason=str(exc))
            raise
        if stage_label:
            print_elapsed_checkpoint(stage_label, bi, n_batches, start_time)
    return np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 384), dtype=np.float32)


def save_partial(vectors_so_far: np.ndarray, chunk_ids_so_far: list[str]) -> None:
    """Write-to-temp-then-replace so a kill mid-write can't corrupt the
    checkpoint (mirrors harvest/01_discover.py's save_checkpoint()).

    The temp path must itself end in ".npy" -- np.save() silently appends
    that suffix to any path that doesn't already have it, which would
    otherwise write e.g. "vectors.partial.npy.tmp.npy" while this function
    tries to replace() the ".tmp"-suffixed name that was never actually
    created."""
    tmp_v = PARTIAL_VECTORS_NPY.parent / (PARTIAL_VECTORS_NPY.stem + ".tmp.npy")
    np.save(tmp_v, vectors_so_far.astype(np.float16))
    tmp_v.replace(PARTIAL_VECTORS_NPY)

    tmp_j = PARTIAL_CHUNK_IDS_JSON.parent / (PARTIAL_CHUNK_IDS_JSON.name + ".tmp")
    with open(tmp_j, "w", encoding="utf-8") as fh:
        json.dump(chunk_ids_so_far, fh)
    tmp_j.replace(PARTIAL_CHUNK_IDS_JSON)


def load_partial_progress(chunk_ids: list[str]) -> tuple[np.ndarray | None, int]:
    """Returns (vectors_so_far, n_done). Only trusts the checkpoint if its
    chunk_ids are an exact prefix of the CURRENT chunks.parquet's chunk_ids
    -- if the corpus changed (re-chunked, reordered) since the partial run
    started, resuming against it would silently misalign vectors to the
    wrong chunks. Starting over is the safe default; (None, 0) means
    "nothing usable to resume"."""
    if not (PARTIAL_VECTORS_NPY.exists() and PARTIAL_CHUNK_IDS_JSON.exists()):
        return None, 0
    try:
        partial_ids = json.loads(PARTIAL_CHUNK_IDS_JSON.read_text(encoding="utf-8"))
        partial_vecs = np.load(PARTIAL_VECTORS_NPY)
        n = len(partial_ids)
        if n == 0 or n != partial_vecs.shape[0]:
            print("[embed] WARNING: partial checkpoint's id count and vector count disagree -- discarding, starting fresh")
            return None, 0
        if partial_ids != chunk_ids[:n]:
            print("[embed] WARNING: partial checkpoint doesn't match the start of the current corpus -- discarding, starting fresh")
            return None, 0
        print(f"[embed] resuming from checkpoint: {n} of {len(chunk_ids)} chunks already embedded")
        return partial_vecs.astype(np.float32), n
    except Exception as exc:
        print(f"[embed] WARNING: couldn't load partial checkpoint ({exc}) -- starting fresh")
        return None, 0


def run_embed_corpus(texts: list[str], chunk_ids: list[str], model, tokenizer, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Like embed_texts(), but checkpoints after every batch to
    PARTIAL_VECTORS_NPY/PARTIAL_CHUNK_IDS_JSON so a crash or interrupt loses
    at most one batch (a few seconds), not the whole multi-hour run. Only
    used by main()'s full-corpus embed -- see embed_texts()'s docstring for
    why the smaller single-query path doesn't need this."""
    resumed_vecs, n_done = load_partial_progress(chunk_ids)
    all_vecs = [resumed_vecs] if resumed_vecs is not None else []
    remaining_texts = texts[n_done:]

    n_batches_total = max(1, -(-len(texts) // batch_size))
    bi = -(-n_done // batch_size)  # count of batches already done, from a resumed checkpoint
    start_time = time.time()
    for i in range(0, len(remaining_texts), batch_size):
        bi += 1
        batch = remaining_texts[i:i + batch_size]
        abs_start = n_done + i + 1
        abs_end = n_done + min(i + batch_size, len(remaining_texts))
        label = f"chunks {abs_start}-{abs_end}"
        try:
            vecs = _embed_batch(batch, model, tokenizer)
            all_vecs.append(vecs)
            progress_line("embed", bi, n_batches_total, label, ok=True)
        except Exception as exc:
            progress_line("embed", bi, n_batches_total, label, ok=False, reason=str(exc))
            raise
        print_elapsed_checkpoint("embed", bi, n_batches_total, start_time)
        save_partial(np.concatenate(all_vecs, axis=0), chunk_ids[:abs_end])

    return np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 384), dtype=np.float32)


def main():
    if not CHUNKS_PARQUET.exists():
        print(f"[embed] ERROR: {CHUNKS_PARQUET} not found. Run 04_chunk.py first.")
        sys.exit(1)

    df = pd.read_parquet(CHUNKS_PARQUET)
    texts = df["text"].tolist()
    chunk_ids = df["chunk_id"].tolist()

    if VECTORS_NPY.exists() and CHUNK_IDS_JSON.exists():
        try:
            existing_ids = json.loads(CHUNK_IDS_JSON.read_text(encoding="utf-8"))
            existing_count = np.load(VECTORS_NPY, mmap_mode="r").shape[0]
            if existing_count == len(chunk_ids) and len(existing_ids) == len(chunk_ids):
                print(f"[embed] SKIP: {VECTORS_NPY} already has {existing_count} vectors "
                      f"matching current chunk count ({len(chunk_ids)}). Delete it to force a re-embed.")
                return
        except Exception:
            pass  # any read/shape issue -> fall through and re-embed

    n_batches = max(1, -(-len(texts) // BATCH_SIZE))
    print(f"[embed] {len(texts)} chunks ({n_batches} batches) to embed with {MODEL_ID} (8-bit ONNX, batch={BATCH_SIZE})")

    model, tokenizer = get_quantized_model_and_tokenizer()

    start = time.time()
    vectors = run_embed_corpus(texts, chunk_ids, model, tokenizer, batch_size=BATCH_SIZE)
    elapsed = time.time() - start

    vectors_f16 = vectors.astype(np.float16)
    np.save(VECTORS_NPY, vectors_f16)
    with open(CHUNK_IDS_JSON, "w", encoding="utf-8") as fh:
        json.dump(chunk_ids, fh)

    # Finished successfully -- the partial checkpoint is now superseded by
    # the real output above, so drop it. Left behind otherwise (crash,
    # Ctrl-C, or a raised exception from run_embed_corpus()) on purpose --
    # that's exactly what the next run resumes from.
    PARTIAL_VECTORS_NPY.unlink(missing_ok=True)
    PARTIAL_CHUNK_IDS_JSON.unlink(missing_ok=True)

    assert vectors_f16.shape[0] == len(chunk_ids), "vector count must match chunk count"

    print(f"\n[embed] DONE in {elapsed:.1f}s ({elapsed / max(len(texts), 1):.3f}s/chunk)")
    print(f"[embed] vectors shape: {vectors_f16.shape} (dtype={vectors_f16.dtype})")
    print(f"[embed] chunk count matches vector count: {vectors_f16.shape[0] == len(chunk_ids)}")
    print(f"[embed] Written: {VECTORS_NPY}, {CHUNK_IDS_JSON}")


if __name__ == "__main__":
    main()
