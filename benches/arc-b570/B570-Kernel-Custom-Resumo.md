# Arc B570 — Kernel Custom: resumo (só v4)

**Data:** 2026-07-17  
**Objetivo:** TG/PP do Bonsai no Intel Arc B570 (Vulkan / Xe2).  
**Estado:** **apenas kernel v4** no tree. Versões v1/v2/v3 e toggles de fallback foram removidos.

---

## 1. Onde está

| O quê | Caminho |
|--------|---------|
| Perfil B570 v4 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (`b570_kernel`) |
| Enable | `ggml_vk_b570_kernel_enabled()` → `INTEL_XE2` |
| Shaders v4 | `mul_mat_vec_q1_0_vec.comp`, `mul_mat_vec_q2_0_vec.comp`, `rms_norm_mul.comp` |
| Doc | `docs/B570-Kernel-v4.md` |
| Bench | `scripts/arc/bench-b570-kernel-v4.ps1` |
| Binários | `build-arc/bin/` (`ggml-vulkan.dll` carrega o perfil) |

---

## 2. Comportamento

No boot Vulkan, em Xe2:

```text
ggml_vulkan: B570 optimized kernel: ON (v4) +mmvq_vec +rmsnorm_mul
```

Sem env vars de versão. O perfil v4 **é** o código no Xe2.

---

## 3. Binários

| O quê | Caminho |
|--------|---------|
| Build | `llama.cpp/build-arc/` |
| CLI / server / bench | `build-arc/bin/*.exe` |
| DLL Vulkan | `build-arc/bin/ggml-vulkan.dll` |

Release GitHub: tag `b570-kernel-v4` no repo `Alguem0001/llama.cpp` (branch `arc-speed`).

---

## 4. Histórico (arquivo, não código)

Experimentos v1–v3 (FA, mmvq LARGE, path id, XL rejeitado) foram **incorporados** na v4 ou descartados.  
Não há mais branches de código para reativar v2/v3.
