# Model Directory

Holds the downloaded GGUF model weight file only (`qwen2.5-1.5b-instruct-q4_k_m.gguf`).
Gitignored — never commit weights.

The ADTC submission template requires `metadata.json` and `download_model.sh` at the **repo
root** (not here) so the profiler can find them — see `../metadata.json` and
`../download_model.sh`. Run `bash ../download_model.sh` (or `bash download_model.sh` from the
repo root) to populate this directory.
