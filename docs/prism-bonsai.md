# PrismML / Bonsai support (on master)

This fork keeps **latest ggml-org/llama.cpp master** and ports PrismML runtime pieces needed
for Bonsai / Ternary-Bonsai without resetting onto the older prism branch.

## Features ported

| Feature | Status | Notes |
|---------|--------|-------|
| Q1_0 | upstream | 1-bit Bonsai |
| Q2_0 group-64 | upstream + CUDA/Vulkan/AVX-VNNI ports | use *_Q2_0_g64.gguf |
| **PQ2_0 group-128** | **ported** | PrismML demo packing; auto-remap when GGUF offsets match g128 |
| CUDA Q2_0/PQ2_0 | ported | MMQ/MMVQ kernels |
| Vulkan Q2_0 | ported | |
| Metal Q2_0 | upstream (g64) | |
| **Hopper wgmma** | **ported (opt-in)** | -DGGML_CUDA_HOPPER_Q1=ON -DGGML_CUDA_CUTLASS_DIR=... + env GGML_HOPPER_Q1; Q1_0 + PQ2_0 |
| **KV mean-centering** | **ported** | --kv-mean-center FNAME with Q4_0 K cache; see docs/kv-mean-center.md |
| **DSpark speculative** | **ported** | arch/model/graph + draft-dspark speculative type; see docs/dspark-scope.md |

## GGUF files

### Ternary (recommended official)

`	ext
*_Q2_0_g64.gguf   -> GGML_TYPE_Q2_0 (group 64)
`

### Ternary (legacy Prism demo)

`	ext
*-Q2_0.gguf       -> auto-detected as GGML_TYPE_PQ2_0 (group 128)
`

Detection: if sequential tensor offsets only match group-128 sizes, the loader remaps
type id 42 (Q2_0) to PQ2_0.

### 1-bit

`	ext
*-Q1_0.gguf
`

## Hopper (optional)

`ash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_HOPPER_Q1=ON -DGGML_CUDA_CUTLASS_DIR=/path/to/cutlass
cmake --build build -j
export GGML_HOPPER_Q1=1   # runtime enable
`

Requires sm_90a (Hopper). Falls through to stock MMQ when unsupported.

## KV mean-centering

`ash
# calibrate
./llama-kv-mean-center -m model.gguf -f calib.txt -o bias.gguf -ctk q4_0
# run
./llama-cli -m model.gguf -ctk q4_0 --kv-mean-center bias.gguf
`

## DSpark

Use draft type draft-dspark with a DSpark GGUF drafter and multi-layer capture on the target.
See docs/dspark-scope.md.

## Branch notes

- master — this integration (latest upstream + ports)
- prism — full PrismML-Eng tree snapshot (older base)
