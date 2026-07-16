# Arc B570 — llama-bench results

**Build:** `1359ad996` (arc-speed) · **Device:** Intel Arc B570 · Vulkan · `-ngl 99 -fa on`

## Ternary-Bonsai 1.7B Q2_0 · pp512 / tg128 · 3 reps

| Mode | Env | pp512 t/s | tg128 t/s |
|------|-----|----------:|----------:|
| **Xe2 FA experiment** (default) | *(unset)* | **6061.58 ± 3.33** | **243.62 ± 0.89** |
| Legacy Intel FA | `GGML_VK_ARC_FA_LEGACY=1` | **6370.55 ± 25.58** | **231.84 ± 0.77** |

### Delta (Xe2 vs Legacy)

| Metric | Δ | Winner |
|--------|--:|--------|
| pp512 | **−4.9%** | Legacy FA |
| tg128 | **+5.1%** | Xe2 FA experiment |

**Interpretation:** On B570 Windows proprietary Vulkan, the Xe2 subgroup FA path helps **token generation** (~5%) but slightly hurts **prompt processing**. For chat (TG-bound) keep default Xe2; for long-context PP-heavy loads set `GGML_VK_ARC_FA_LEGACY=1`.

Device caps reported: `fp16:1 int_dot:1 matrix cores: KHR_coopmat · warp 32 · shmem 48KB`.

## Bonsai-27B Q1_0 · pp512 / tg128 · 2 reps

| Mode | pp512 t/s | tg128 t/s |
|------|----------:|----------:|
| Xe2 FA (default) | 474.66 ± 1.73 | 36.65 ± 0.04 |
| Legacy FA | 478.09 ± 0.16 | 36.41 ± 0.11 |

### Delta (Xe2 vs Legacy on 27B)

| Metric | Δ | Note |
|--------|--:|------|
| pp512 | ~−0.7% | noise / within margin |
| tg128 | ~+0.7% | noise / within margin |

On the **real Bonsai-27B Q1_0** workload, FA path choice is **neutral** (bandwidth-bound decode). Prefer keeping Xe2 default; use Legacy only if you measure PP regression on your prompts.

## SYCL vs Vulkan — Bonsai-27B Q1_0 (2 reps)

| Backend | pp512 t/s | tg128 t/s |
|---------|----------:|----------:|
| **Vulkan** (build-arc) | **~475** | **~36.6** |
| SYCL / Level Zero (build-sycl) | ~416 | ~15.9 |

**Winner on Arc B570 Windows for Bonsai Q1_0: Vulkan** (~2.3× TG, ~1.14× PP).

Notes:
- SYCL **does not support Q2_0** (`unsupport data type=q2_0`) — 1.7B Ternary cannot run on this SYCL build.
- SYCL needs `setvars.bat` + `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` at runtime.
- Keep daily path on **Vulkan** (`build\bin` synced from `build-arc`).

## Env knobs (this fork)

| Variable | Effect |
|----------|--------|
| `GGML_VK_ARC_FA_LEGACY=1` | Upstream Intel FA (subgroups off) |
| `GGML_VK_ARC_MMVQ_WG=large\|subgroup` | Force mmvq workgroup size |
| `GGML_VK_FORCE_MMVQ=1` | Force mmvq path |
| `GGML_VK_DISABLE_MMVQ=1` | Disable mmvq |
