# Arc B570 — Kernel v4 (único perfil)

**Data:** 2026-07-17  
**Branch:** `arc-speed`  
**Regra:** **só existe kernel v4**. Não há fallback para v2/v3/off no código nem env de seleção de versão.

---

## O que é a v4

Perfil fixo em **Intel Xe2 / B570** (`architecture == INTEL_XE2`):

| Bloco | Comportamento |
|--------|----------------|
| FA TG | subgroup SIMD16 + WG 64 no decode (`n_rows==1`) |
| MMQ small tile | BK aprofundado (reuse de K) |
| mmvq Q1_0/Q2_0 | workgroup **LARGE** no decode (path principal + id) |
| mmvq thresholds | limiares M mais altos no Xe2 |
| `mul_mat_vec_q{1,2}_0_vec.comp` | shaders vetorizados (pipelines sempre criados no Xe2) |
| `rms_norm_mul.comp` | fusão RMS_NORM→MUL + `add_rms_fusion` permitido no Intel Xe2 |

Em GPUs que não são Xe2 o perfil B570 **não** liga (código upstream normal).

---

## Código

| Item | Onde |
|------|------|
| Flag única `b570_kernel` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (struct device) |
| Enable | `ggml_vk_b570_kernel_enabled()` → `arch == INTEL_XE2` |
| Log boot | `ggml_vulkan: B570 optimized kernel: ON (v4) +mmvq_vec +rmsnorm_mul` |
| Shaders | `vulkan-shaders/mul_mat_vec_q1_0_vec.comp`, `q2_0_vec.comp`, `rms_norm_mul.comp` |
| Register gen | `vulkan-shaders-gen.cpp` |

**Removido de propósito:**

- `GGML_VK_B570_KERNEL=0|1|2|3` e parse multi-modo  
- `GGML_VK_B570_MMVQ_VEC` / `GGML_VK_B570_FUSE_RMSNORM_MUL` como toggles  
- `GGML_VK_B570_MMVQ_XL` e `DMMV_WG_SIZE_XL`  
- `GGML_VK_ARC_FA_LEGACY` como desliga-kernel  
- benches/scripts A/B de versões antigas  

---

## Bench

```powershell
powershell -File scripts/arc/bench-b570-kernel-v4.ps1
powershell -File scripts/arc/bench-b570-kernel-v4.ps1 -Phase matrix
```

---

## Backlog (não é “versão de kernel”)

1. Coopmat PP no B570  
2. MMVQ2+SwiGLU fundido  
3. Fusão same-activation mais ampla  
