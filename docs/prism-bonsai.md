# PrismML / Bonsai support (on master)

This fork keeps **latest `ggml-org/llama.cpp` master** and ports the PrismML pieces
needed to run Ternary-Bonsai models **without** resetting onto the older `prism` branch.

## What is already upstream (master)

| Backend | Q1_0 (1-bit) | Q2_0 ternary (group-64) |
|---------|--------------|-------------------------|
| CPU     | yes          | yes (ARM NEON + generic) |
| Metal   | yes          | yes |
| CUDA    | yes          | **ported here** (from [ggml-org#25707](https://github.com/ggml-org/llama.cpp/pull/25707)) |
| Vulkan  | yes          | **ported here** (from [ggml-org#25430](https://github.com/ggml-org/llama.cpp/pull/25430)) |

Extra on this fork:

- **x86 AVX-VNNI / AVX-512-VNNI** fast path for Q2_0 (adapted from PrismML for **group-64**)

## GGUF files to use

Official master layout is **group size 64** (`QK2_0 = 64`).

Use Hugging Face files named `*_Q2_0_g64.gguf` (or the renamed plain `Q2_0` once PrismML migrates):

```text
hf download prism-ml/Ternary-Bonsai-8B-gguf  Ternary-Bonsai-8B-Q2_0_g64.gguf  --local-dir models
hf download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0_g64.gguf --local-dir models
```

**Do not** use the older demo files `*-Q2_0.gguf` with **group-128** packing from the PrismML
fork — those need the `prism` branch / prebuilt PrismML binaries (same type id, different block size).

1-bit Bonsai (`Q1_0`) works on stock master backends.

## Not ported (intentionally)

These remain only on [PrismML-Eng/llama.cpp `prism`](https://github.com/PrismML-Eng/llama.cpp/tree/prism)
because they are large or tightly coupled to that tree:

- CUDA Hopper wgmma optional path
- DSpark speculative drafter
- KV mean-centering tooling
- Q2_0 **group-128** type layout

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # add -DGGML_CUDA=ON / -DGGML_VULKAN=ON as needed
cmake --build build -j
```
