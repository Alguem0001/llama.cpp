# Intel Arc B570 + Bonsai — relatório completo (comportamento, kernel, código, benches)

**Data:** 2026-07-16  
**Branch:** `arc-speed` (`Alguem0001/llama.cpp`)  
**Arquivo-fonte do kernel:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`  
**Logs brutos:** `benches/arc-b570/`

---

## 1. Setup

| Item | Valor |
|------|--------|
| GPU | Intel Arc **B570** (Battlemage, **Xe2**, SIMD16) |
| Driver | Vulkan Windows proprietário (~101.x) |
| Caps (llama-bench) | `fp16:1` · `int_dot:1` · `matrix cores: KHR_coopmat` · warp **32** · shmem **48 KB** · UMA=0 |
| Backend diário | **Vulkan** (`build-arc` → `build\bin`) |
| Backend testado (2º) | SYCL / Level Zero (`build-sycl`, oneAPI 2025.1) |

### Modelos

| Arquivo | Tamanho | Uso |
|---------|--------:|-----|
| `Bonsai-27B-Q1_0.gguf` | ~3.54 GB | **alvo principal** (27B, 1-bit) |
| `Bonsai-27B-mmproj-Q8_0.gguf` | ~0.59 GB | visão (não nos benches TG) |
| `Ternary-Bonsai-1.7B-Q2_0_g64.gguf` | ~0.46 GB | proxy rápido p/ ver se tuning “aparece” |

Flags padrão de bench/servidor:

```text
-ngl 99 -fa on -p 512 -n 128
# launcher WinUI também:
-b 512 -ub 256
```

---

## 2. Como a B570 se comporta com o Bonsai-27B

### 2.1 O que o decode pede da GPU

1. **Token generation (TG)** — cada token re-lê praticamente todos os pesos → **bandwidth-bound**.  
   Com Q1_0 o GGUF já é pequeno; o t/s ≈ `bytes_por_token / banda_útil`.
2. **Prompt processing (PP)** — matmul grande + FA em batch → mais **compute** (int-dot, coopmat, FA).
3. **Consequência:** no 27B Q1_0, micro-tuning de workgroup/FA **quase não move o TG** (~36 t/s). No 1.7B ainda há folga e os % aparecem.

### 2.2 Números “reais” do 27B (Vulkan, estáveis)

| Métrica | Ordem de grandeza |
|---------|-------------------|
| **pp512** | ~**470–480 t/s** |
| **tg128** | ~**36 t/s** |

Isso é o comportamento esperado do **Bonsai-27B-Q1_0** nesta B570 com o stack atual — não indica bug se estiver estável.

### 2.3 1.7B como contraste

| Métrica | 1.7B Q2_0 |
|---------|----------:|
| pp512 | ~**6 000–6 400 t/s** |
| tg128 | ~**226–244 t/s** |

Aqui diferenças de FA/kernel de alguns % **aparecem**.

---

## 3. O que tentamos (cronológico)

### 3.1 Vulkan vs SYCL (oneAPI)

- Instalado **oneAPI Base Toolkit 2025.1**.
- Build `build-sycl` (com patches de arch: remover `bmg_g31` / `wcl` ausentes nos headers 2025.1).
- SYCL **não suporta Q2_0** (`unsupport data type=q2_0`).
- No **27B Q1_0**:

| Backend | pp512 | tg128 |
|---------|------:|------:|
| **Vulkan** | ~475 | **~36.6** |
| SYCL / Level Zero | ~416 | **~15.9** |

**Vulkan ~2.3× em TG.** Uso diário: Vulkan.

### 3.2 Flash Attention — Xe2 vs Legacy Intel

Upstream desliga subgroups em **todo** Intel (histórico Xe1). Testamos path mais “Xe2”:

| Modelo | Path | pp512 | tg128 |
|--------|------|------:|------:|
| 1.7B | Xe2 FA | 6062 | **244** |
| 1.7B | Legacy | **6371** | 232 |
| 27B | Xe2 | 475 | 36.7 |
| 27B | Legacy | 478 | 36.4 |

- **1.7B:** Xe2 **+5% TG**, **−5% PP**.  
- **27B:** empate (ruído).

### 3.3 Perfil “kernel B570” (seleção de pipelines Vulkan)

Não é um `.comp` GLSL novo isolado. No llama.cpp Vulkan o ganho prático é um **perfil de seleção** de kernels já existentes (FA scalar, mmvq, warptiles MMQ), ligado por env:

```text
GGML_VK_B570_KERNEL=0   → normal (upstream-like)
GGML_VK_B570_KERNEL=1   → B570 v2
# default em INTEL_XE2: ON
```

#### v1 (agressivo) — **regressão**

- Warptile small com **128 threads**
- Forçar `mmvq_mode=1`
- FA `block_cols=128`
- LARGE em quase tudo  

**1.7B TG: 230 → 205 (−11%)** → descartado.

#### v2 (conservador) — **o que ficou**

| Peça | Normal | B570 v2 |
|------|--------|---------|
| FA no decode (`n_rows==1`) | subgroups off, sg=32 | **subgroups on, sg=16, wg=64** |
| FA no PP | subgroups off | igual |
| mmvq Q1_0/Q2_0 decode | heurística Intel | **LARGE** |
| MMQ tile small BK | 32 | **64** |
| force `mmvq_mode` | não | **não** |

| Modelo | Normal (pp / tg) | B570 v2 (pp / tg) | Δ TG |
|--------|------------------|-------------------|------|
| 1.7B Q2_0 | 6303 / 226.5 | 6319 / **229.2** | **+1.2%** |
| **27B Q1_0** | 469 / **36.0** | 466 / **35.9** | **~0%** |

### 3.4 Outras alavancas

- Launchers WinUI: `-fa on -b 512 -ub 256`, Import GGUF, HF download, UI draft/speculative (`-md`, `--draft-max`).
- Draft 1.7B Ternary **não** é draft confiável para 27B Bonsai (família/tokenizer).
- Ports Prism (Q1_0, etc.) **habilitam** o modelo; não “aceleram a B570” além de pesos menores.

---

## 4. Tabela resumo: o que funcionou

| Tentativa | Veredito |
|-----------|----------|
| Vulkan + `-fa on` + full offload | **Base boa** |
| SYCL no 27B Q1_0 | **Pior** (~16 vs ~36 tg) |
| FA Xe2 no 1.7B | **+TG / −PP** |
| FA Xe2 no 27B | **Neutro** |
| Kernel B570 v1 | **Regressão** (−11% TG 1.7B) |
| Kernel B570 v2 | **+1% TG 1.7B**, **0% no 27B** |
| Forçar mmvq / warptile 128 thr / Bc=128 | **Dói** |
| Draft 1.7B→27B | **Não validado** |

### Diagrama mental

```text
Bonsai-27B Q1_0 na B570
├─ Decode ~36 t/s  →  limitado por BANDA de memória
├─ Prompt ~475 t/s →  compute + FA; Vulkan OK
├─ SYCL              →  bem pior em TG (Windows)
├─ Micro-kernel FA/mmvq
│   ├─ 1.7B: ± alguns %
│   └─ 27B: ruído (~0%)
└─ Próximo salto de TG no 27B
    └─ draft/speculative da mesma família, quant menor, ou banda —
       não “WG mágico”
```

---

## 5. Como repetir os benches

```powershell
cd "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\llama.cpp"

# Normal
$env:GGML_VK_B570_KERNEL = "0"
.\build-arc\bin\llama-bench.exe `
  -m ".\models\..\models\Bonsai-27B-Q1_0.gguf" `
  -ngl 99 -fa on -p 512 -n 128 -r 3

# Kernel B570
$env:GGML_VK_B570_KERNEL = "1"
# mesmo comando

# Script A/B
powershell -File scripts\arc\bench-b570-kernel.ps1 -Also27B
```

Logs no console ao iniciar o backend:

```text
ggml_vulkan: B570 optimized kernel: ON (v2)  [toggle: GGML_VK_B570_KERNEL=0|1]
# ou
ggml_vulkan: B570 optimized kernel: OFF (normal)  [toggle: GGML_VK_B570_KERNEL=0|1]
```

### Variáveis de ambiente úteis

| Variável | Efeito |
|----------|--------|
| `GGML_VK_B570_KERNEL=0\|1` | Normal vs perfil B570 v2 (default ON em Xe2) |
| `GGML_VK_ARC_FA_LEGACY=1` | FA Intel legacy (também desliga B570 FA path) |
| `GGML_VK_ARC_MMVQ_WG=large\|subgroup` | Força workgroup mmvq |
| `GGML_VK_FORCE_MMVQ=1` / `DISABLE` | Força / desliga mmvq |
| `GGML_VK_VISIBLE_DEVICES=0` | Primeiro device Vulkan |

---

## 6. Código do kernel B570 (completo, como no tree)

**Arquivo:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`  
**Nota:** o “kernel” é um **perfil de seleção/tunagem** em C++ sobre shaders SPIR-V já gerados (`flash_attn*.comp`, `mul_mat_vec*.comp`, `mul_mmq.comp`, etc.). Não há um único arquivo `.comp` “B570.comp”; o código abaixo é **todo** o que implementa o perfil.

### 6.1 Detecção de arquitetura Xe2 (upstream + base do perfil)

```cpp
// ggml-vulkan.cpp — detecção INTEL_XE2 (min subgroup size == 16)
if (subgroup_size_control_props.minSubgroupSize == 16) {
    // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
    return vk_device_architecture::INTEL_XE2;
} else if (subgroup_size_control_props.minSubgroupSize == 8 &&
         integer_dot_product && integer_dot_props.integerDotProduct4x8BitPackedSignedAccelerated) {
    return vk_device_architecture::INTEL_XE1;
}
```

Device ID B570 (contagem de cores / EU helper):

```cpp
case 0xE20C:  // B570
    return 18;
case 0xE20B:  // B580
case 0xE211:  // Pro B60
    return 20;
```

### 6.2 Flag no device

```cpp
// em struct vk_device (aprox. linhas 748–750)
// ARC-SPEED: Battlemage B570/Xe2 optimized kernel profile (FA + mmvq + warptiles)
// Toggle: GGML_VK_B570_KERNEL=0|1|off|on  (default ON when architecture==INTEL_XE2)
bool b570_kernel {};
```

### 6.3 Ativação via env + log

```cpp
// B570 / Xe2 kernel profile — default ON for INTEL_XE2; override with GGML_VK_B570_KERNEL=0|1
static bool ggml_vk_b570_kernel_from_env(vk_device_architecture arch) {
    const char * e = getenv("GGML_VK_B570_KERNEL");
    if (e != nullptr) {
        if (e[0] == '0' || strcmp(e, "off") == 0 || strcmp(e, "OFF") == 0 ||
            strcmp(e, "normal") == 0 || strcmp(e, "NORMAL") == 0) {
            return false;
        }
        if (e[0] == '1' || strcmp(e, "on") == 0 || strcmp(e, "ON") == 0 ||
            strcmp(e, "opt") == 0 || strcmp(e, "OPT") == 0) {
            return true;
        }
    }
    // Legacy env still forces upstream FA path (implies no B570 kernel for FA)
    if (getenv("GGML_VK_ARC_FA_LEGACY") != nullptr) {
        return false;
    }
    return arch == INTEL_XE2;
}
```

```cpp
// na inicialização do device
device->b570_kernel = ggml_vk_b570_kernel_from_env(device->architecture);
// Do NOT force mmvq_mode=1 — that regressed TG in B570 v1 benches.

std::cerr << "ggml_vulkan: B570 optimized kernel: "
          << (device->b570_kernel ? "ON (v2)" : "OFF (normal)")
          << "  [toggle: GGML_VK_B570_KERNEL=0|1]" << std::endl;
```

### 6.4 Flash Attention scalar — tuning B570 v2

Função completa `get_fa_tuning_params_scalar` (trechos B570 marcados nos comentários):

```cpp
static vk_fa_tuning_params get_fa_tuning_params_scalar(
    const vk_device& device,
    uint32_t hsk, uint32_t hsv,
    uint32_t n_rows, uint32_t n_kv,
    ggml_type k_type, ggml_type v_type,
    bool f32acc)
{
    vk_fa_tuning_params result{};
    result.path = FA_SCALAR;
    const bool b570 = device->b570_kernel;

    if (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Upstream: disable subgroups on all Intel (Xe1 history).
        // B570 v2: on TG (n_rows==1) enable SIMD16 subgroups (measured +5% TG on 1.7B);
        // on PP keep upstream (subgroups off) to protect prompt speed.
        if (b570 && n_rows == 1) {
            result.subgroup_size = 16;
            result.disable_subgroups = false;
        } else {
            result.subgroup_size = 32;
            result.disable_subgroups = true;
        }
    } else if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        result.subgroup_size = n_rows < 4 ? 32 : device->subgroup_size;
    } else {
        result.subgroup_size = device->subgroup_size;
    }

    uint32_t row_split_max_hsk = 64;
    if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        row_split_max_hsk = n_rows <= 8 ? 64 : 128;
    }
    result.row_split = (n_rows < 4 || hsk <= row_split_max_hsk) ? 1 : 4;

    if (result.subgroup_size > 32 && (n_rows < 4 || hsk < (result.row_split == 1 ? 128 : 64))) {
        result.workgroup_size = result.subgroup_size * 2;
    } else {
        result.workgroup_size = result.subgroup_size * 4;
    }
    // B570 TG: 16*4 = 64 threads
    if (b570 && n_rows == 1 && !result.disable_subgroups) {
        result.workgroup_size = 64;
    }

    const uint32_t D = hsk | hsv;

    const bool intel_reduce_rows = device->vendor_id == VK_VENDOR_ID_INTEL;
    const bool reduce_block_rows = D & 8 || n_kv < 1024 || intel_reduce_rows;

    if (n_rows == 1) {
        result.block_rows = 1;
        result.block_cols = 64; // keep 64 — Bc=128 regressed on B570
    } else {
        if (result.row_split == 1) {
            result.block_rows = n_rows == 2 ? 2 : ((n_rows <= 4 || reduce_block_rows) ? 4 : 8);
        } else {
            result.block_rows = n_rows <= 4 ? 4 : ((n_rows <= 8 || reduce_block_rows) ? 8 : 16);
        }
        result.block_cols = (D & 8) ? 64 : 32;
    }

    const uint32_t D_lsb = D ^ (D & (D-1));
    result.d_split = std::min(std::min(result.subgroup_size, 8u), D_lsb / 4);
    result.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    if (!reduce_block_rows &&
        !ggml_vk_flash_attn_scalar_shmem_support(device, result, hsk, hsv, f32acc, k_type, v_type)) {
        result.block_rows /= 2;
    }

    // ... AMD RDNA limit_occupancy_shmem omitted (não B570) ...

    return result;
}
```

**Shaders FA usados em runtime** (gerados de GLSL em `ggml/src/ggml-vulkan/vulkan-shaders/`):  
`flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp`, etc. — o perfil B570 só muda **spec constants / workgroup / path** ao criar o pipeline.

### 6.5 Warptile MMQ (tile small) — B570 v2

```cpp
l_warptile_mmq = { 128,             128, 128, 32, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 };
m_warptile_mmq = { 128,              64,  64, 32, subgroup_size_8,     32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
s_warptile_mmq = { subgroup_size_32, 32,  32, 32, s_warptile_wm,       32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

// B570 v2: only deepen BK on small MMQ tile (more K reuse) without inflating threads
// (full 128-thread s_warptile regressed TG ~11% in v1).
if (device->b570_kernel) {
    s_warptile_mmq = { subgroup_size_32, 32, 32, 64, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };
}
```

Layout típico do vector warptile:  
`{ threads, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, subgroup_size }` (ver comentários no arquivo).

**v1 descartado** (por referência — **não** está no tree atual):

```cpp
// NÃO USAR — regrediu TG ~11% no 1.7B
// s_warptile     = { 128, 64, 64, 16, sg16, 32, 2, tm_s, tn_s, tk_s, sg16 };
// s_warptile_mmq = { 128, 64, 64, 64, sg16, 32, 2, tm_s, tn_s, tk_s, sg16 };
// force mmvq_mode = 1;
// FA block_cols = 128;
```

### 6.6 mmvq / dequant mul_mat_vec — workgroup LARGE

#### Path principal (não-id)

```cpp
// heuristic to choose workgroup size
uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
// optional force — GGML_VK_ARC_MMVQ_WG=large|subgroup
const char * arc_mmvq_wg = getenv("GGML_VK_ARC_MMVQ_WG");
const bool force_large_wg = arc_mmvq_wg && (strcmp(arc_mmvq_wg, "large") == 0 || strcmp(arc_mmvq_wg, "LARGE") == 0);
const bool force_sub_wg   = arc_mmvq_wg && (strcmp(arc_mmvq_wg, "subgroup") == 0 || strcmp(arc_mmvq_wg, "SUBGROUP") == 0);
const bool b570 = ctx->device->b570_kernel;

if (force_large_wg) {
    dmmv_wg = DMMV_WG_SIZE_LARGE;
} else if (force_sub_wg) {
    dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
} else if ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) ||
           ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
    // B570 v2: slightly higher m threshold for LARGE (more EU occupancy on Xe2)
    const uint32_t m_lim = b570 ? 12288u : 8192u;
    const uint32_t m_q6  = b570 ? 6144u  : 4096u;
    if (a_type == GGML_TYPE_Q6_K) {
        if (m < m_q6 && k >= 1024) {
            dmmv_wg = DMMV_WG_SIZE_LARGE;
        }
    } else {
        if (m <= m_lim && k >= 1024) {
            dmmv_wg = DMMV_WG_SIZE_LARGE;
        }
    }
    // Bonsai quants: always LARGE on decode
    if (b570 && (a_type == GGML_TYPE_Q1_0 || a_type == GGML_TYPE_Q2_0) && m <= 8 && k >= 512) {
        dmmv_wg = DMMV_WG_SIZE_LARGE;
    }
}

if (b_type == GGML_TYPE_Q8_1) {
    // Upstream forces SUBGROUP on all Intel (keep that; LARGE on Q8_1 regressed)
    if (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && !force_large_wg) {
        dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
    }
    return ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][num_cols-1];
}

return b_type == GGML_TYPE_F32
    ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][num_cols-1]
    : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][num_cols-1];
```

#### Path mul_mat_vec **id** (MoE / expert)

```cpp
uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
const char * arc_mmvq_wg_id = getenv("GGML_VK_ARC_MMVQ_WG");
const bool b570_id = ctx->device->b570_kernel;
if (arc_mmvq_wg_id && (strcmp(arc_mmvq_wg_id, "large") == 0 || strcmp(arc_mmvq_wg_id, "LARGE") == 0)) {
    dmmv_wg = DMMV_WG_SIZE_LARGE;
} else if (arc_mmvq_wg_id && (strcmp(arc_mmvq_wg_id, "subgroup") == 0 || strcmp(arc_mmvq_wg_id, "SUBGROUP") == 0)) {
    dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
} else if ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) ||
           ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
    const uint32_t m_lim = b570_id ? 12288u : 8192u;
    const uint32_t m_q6  = b570_id ? 6144u  : 4096u;
    if (a_type == GGML_TYPE_Q6_K) {
        if (m < m_q6 && k >= 1024) {
            dmmv_wg = DMMV_WG_SIZE_LARGE;
        }
    } else {
        if (m <= m_lim && k >= 1024) {
            dmmv_wg = DMMV_WG_SIZE_LARGE;
        }
    }
}

if (b_type == GGML_TYPE_Q8_1) {
    if (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
    }
    return ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
}

return ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
```

`DMMV_WG_SIZE_SUBGROUP` vs `DMMV_WG_SIZE_LARGE` mapeiam para workgroups  
`subgroup_size` vs `subgroup_size * 4` na criação dos pipelines mmvq (ver loop `for (uint32_t w = 0; w < DMMV_WG_SIZE_COUNT; ++w)` no mesmo arquivo).

### 6.7 Shaders SPIR-V relacionados (não alterados; só selecionados)

Gerados a partir de `ggml/src/ggml-vulkan/vulkan-shaders/`:

| Família | Exemplos `.comp` |
|---------|------------------|
| Flash attention | `flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp`, `flash_attn_mask_opt.comp` |
| Mul-mat vec / dequant | `mul_mat_vec.comp`, `mul_mat_vec_q*.comp`, `mul_mat_vec_iq*.comp`, `dequant_q1_0.comp`, `dequant_q2_0.comp` |
| MMQ | `mul_mmq.comp`, `mul_mm.comp`, `mul_mm_cm2.comp` |

O perfil B570 **não reescreve** esses GLSL; muda **quando** e **com quais specialization constants / workgroup sizes** eles rodam.

---

## 7. Conclusões práticas

1. **Use Vulkan** (`build\bin` / `build-arc`), não SYCL, para Bonsai-27B nesta máquina.  
2. **~36 t/s** no 27B Q1_0 é o comportamento esperado (decode bound por banda).  
3. **Kernel B570 v2** pode ficar **ligado** (default Xe2): não atrapalha o 27B e ajuda ~1% no 1.7B.  
4. Chat **muito prompt-heavy:** teste `GGML_VK_B570_KERNEL=0` ou `GGML_VK_ARC_FA_LEGACY=1`.  
5. **Mais velocidade no 27B:** draft GGUF da **mesma família** + speculative, ou aceitar o teto de banda — não esperar 2× de micro-heurística mmvq.

---

## 8. Referências no repo

| Caminho | Conteúdo |
|---------|----------|
| `docs/b570-bonsai-kernel-report.md` | **Este relatório** |
| `docs/arc-speed.md` | Playbook Arc / scripts |
| `benches/arc-b570/RESULTS.md` | Tabelas de bench |
| `benches/arc-b570/kernel-v2-*.txt` | Outputs brutos A/B |
| `scripts/arc/bench-b570-kernel.ps1` | Script A/B |
| `scripts/arc/build-vulkan.ps1` | Build Vulkan |
| `scripts/arc/build-sycl.ps1` | Build SYCL |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | **Código-fonte do perfil** |

**Branch:** `https://github.com/Alguem0001/llama.cpp/tree/arc-speed`  
**Repo sandbox:** `https://github.com/Alguem0001/llama.cpp-arc-speed`
