# Intel Arc speed experiments (B570 / Battlemage Xe2)

Repo branch: **`arc-speed`** · remote: [Alguem0001/llama.cpp-arc-speed](https://github.com/Alguem0001/llama.cpp-arc-speed)

This tree starts from our Prism/Bonsai master and adds **Arc-focused** build scripts,
runtime presets, and a small Vulkan flash-attention experiment for **Xe2**.

## Hardware context

| Item | Your setup |
|------|------------|
| GPU | Intel Arc **B570** (Battlemage, Xe2, SIMD16) |
| Backend in daily use | **Vulkan** (Windows proprietary driver) |
| Model class | Small quants (e.g. Bonsai Q1_0) → **memory-bandwidth bound** on decode |

TG (token generation) is almost always bandwidth-bound. PP (prompt) is more compute-bound
and benefits from integer-dot / coopmat / flash-attn paths.

## What usually helps on Arc (ordered)

### 1. Runtime flags (no rebuild)

```text
llama-server ...
  -ngl 99                 # all layers on GPU
  -fa on                  # flash attention (critical)
  -c 8192                 # only as large as you need
  -b 512 -ub 256          # batch / ubatch (tune PP)
  --no-mmap               # sometimes faster cold load on Windows
  -t 8                    # CPU threads for residual work (match P-cores)
```

Optional env vars (Vulkan backend):

| Variable | Effect |
|----------|--------|
| `GGML_VK_DISABLE_COOPMAT=1` | A/B if coopmat is slower on your driver |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | A/B int-dot path |
| `GGML_VK_ARC_FA_LEGACY=1` | **This fork:** force upstream Intel FA tuning (subgroups off) |
| `GGML_VK_VISIBLE_DEVICES=0` | Force first Vulkan device |

### 2. Quant choice

| Goal | Prefer |
|------|--------|
| Max TG on Arc | Smaller / bandwidth-friendly quants (Q4_K_M, Q3_K, IQ4, **Q1_0 Bonsai**) |
| Max quality at same TG | Stay on small weights; avoid Q8 unless PP matters more |
| Vision | Keep mmproj Q8; main weights stay small |

Decode ≈ `bytes_per_token / effective_bandwidth`. Halving model size ≈ ~2× TG if still bandwidth-bound.

### 3. Speculative decoding

If you have a tiny draft GGUF of the same family:

```text
llama-server -m main.gguf -md draft.gguf --draft-max 16 ...
```

On Arc this often helps **40–90%** when draft is cheap and acceptance is good (see community reports 2026).

### 4. SYCL / oneAPI (optional second backend)

Not installed by default on this machine. When oneAPI is present:

```powershell
# after setvars.bat
cmake -B build-sycl -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --config Release -j
```

SYCL can beat Vulkan on some Arc SKUs for TG; Vulkan is still the best “works today” path on Windows without oneAPI.

### 5. Driver / system

- Keep **Intel Arc GPU driver** current (your Vulkan driver reported ~101.x).
- Resizable BAR / Above 4G decoding ON in firmware when available.
- Prefer **exclusive** GPU use (no heavy desktop 3D while benching).
- Power plan: High performance; avoid iGPU stealing the default adapter (pin device with `--device` / `GGML_VK_VISIBLE_DEVICES`).

## Code experiment in this branch

**File:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — flash-attn scalar tuning.

- **Upstream:** all Intel → `disable_subgroups=true`, smaller block rows.
- **arc-speed on Xe2:** allow subgroups (SIMD16), allow larger block rows.
- **Rollback without rebuild:** `set GGML_VK_ARC_FA_LEGACY=1`

Always A/B with `llama-bench` before declaring a win.

## Build (Vulkan, optimized)

```powershell
# from repo root
powershell -File scripts/arc/build-vulkan.ps1
```

Or manually:

```powershell
cmake -B build-arc -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_VULKAN=ON `
  -DGGML_NATIVE=ON `
  -DGGML_LTO=ON
cmake --build build-arc -j
```

## Bench

```powershell
powershell -File scripts/arc/bench.ps1 -Model "C:\path\to\model.gguf"
```

Compare:

1. stock flags  
2. `-fa on`  
3. with/without `GGML_VK_ARC_FA_LEGACY=1`  
4. different `-b` / `-ub`

Record **pp512** and **tg128** (or tg64) with 3 reps.

## Roadmap (this repo)

- [x] Arc docs + build/bench scripts  
- [x] Xe2 flash-attn tuning experiment + env rollback  
- [ ] Mul-mat / mmvq workgroup heuristics for Xe2 (next)  
- [ ] Optional SYCL dual-build script when oneAPI present  
- [ ] Draft-model speculative preset for Bonsai sizes  
- [ ] Capture `llama-bench` baseline JSON under `benches/arc-b570/`

## Relation to other repos

| Repo | Role |
|------|------|
| `Alguem0001/llama.cpp` | Daily driver: Prism/Bonsai correctness + features |
| **`Alguem0001/llama.cpp-arc-speed`** | Speed sandbox for Arc (may diverge / break) |
| `Alguem0001/Bansai-WinUI` | Launcher UI |

Promote patches that win benches from **arc-speed → main llama.cpp**.
